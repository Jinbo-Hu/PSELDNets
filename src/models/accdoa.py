import torch
import torch.nn as nn

from models.components.conformer import ConformerBlocks
from models.components.model_utilities import Decoder, get_conv2d_layer
from models.components.backbone import CNN8, CNN12
from models.components.htsat import HTSAT_Swin_Transformer
from models.components.passt import PaSST
from models.components.utils import interpolate


class CRNN(nn.Module):
    def __init__(self, cfg, num_classes, in_channels=7, encoder='CNN8', pretrained_path=None,
                 audioset_pretrain=True, num_features=[32, 64, 128, 256]):
        super().__init__()

        data = cfg.data
        mel_bins = cfg.data.n_mels
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.label_res = 0.1
        self.interpolate_time_ratio = 2 ** 3
        self.output_frames = None #int(data.train_chunklen_sec / 0.1)
        self.pred_res = int(data.sample_rate / data.hoplen * self.label_res) # 10
        
        self.scalar = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(in_channels)])
        if encoder == 'CNN8':
            self.convs = CNN8(in_channels, num_features)
        elif encoder == 'CNN12':
            self.convs = CNN12(in_channels, num_features)
            if pretrained_path:
                print('Loading pretrained model from {}...'.format(pretrained_path))
                self.load_ckpts(pretrained_path, audioset_pretrain)
        else:
            raise NotImplementedError(f'encoder {encoder} is not implemented')
        
        self.num_features = num_features

        self.decoder = Decoder(cfg.model.decoder, num_features[-1], 
                               num_layers=cfg.model.num_decoder_layers)
        self.fc = nn.Linear(num_features[-1], 3*num_classes, bias=True, )
        self.final_act = nn.Tanh()
    
    def load_ckpts(self, pretrained_path, audioset_pretrain=True):
        if audioset_pretrain:
            CNN14_ckpt = torch.load(pretrained_path, map_location='cpu')['model']
            CNN14_ckpt['conv_block1.conv1.weight'] = nn.Parameter(
                CNN14_ckpt['conv_block1.conv1.weight'].repeat(1, self.in_channels, 1, 1) / self.in_channels)
            missing_keys, unexpected_keys = self.convs.load_state_dict(CNN14_ckpt, strict=False)
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
            for ich in range(self.in_channels):
                self.scalar[ich].weight.data.copy_(CNN14_ckpt['bn0.weight'])
                self.scalar[ich].bias.data.copy_(CNN14_ckpt['bn0.bias'])
                self.scalar[ich].running_mean.copy_(CNN14_ckpt['bn0.running_mean'])
                self.scalar[ich].running_var.copy_(CNN14_ckpt['bn0.running_var'])
                self.scalar[ich].num_batches_tracked.copy_(CNN14_ckpt['bn0.num_batches_tracked'])
        else:
            ckpt = torch.load(pretrained_path, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for key, value in self.state_dict().items():
                if key.startswith('fc.'): print(f'Skipping {key}...')
                else: value.data.copy_(ckpt[key])

    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, time_frames, mel_bins)
        """

        N, _, T, _ = x.shape
        self.output_frames = int(T // self.pred_res)

        # Compute scalar
        x = x.transpose(1, 3)
        for nch in range(x.shape[-1]):
            x[..., [nch]] = self.scalar[nch](x[..., [nch]])
        x = x.transpose(1, 3)

        # encoder
        x = self.convs(x)
        x = x.mean(dim=3) # (N, C, T)
        
        # decoder
        x = x.permute(0, 2, 1) # (N, T, C)
        x = self.decoder(x) # (N, T, C)
        
        x = interpolate(x, ratio=self.interpolate_time_ratio) # (N, T, C)
        x = x.reshape(N, self.output_frames, self.pred_res, -1).mean(dim=2) # (N, T, C)

        # fc
        x = self.final_act(self.fc(x))
        
        return {
            'accdoa': x,
        }


class ConvConformer(CRNN):
    def __init__(self, cfg, num_classes, in_channels=7, encoder='CNN8', pretrained_path=None, 
                 audioset_pretrain=True, num_features=[32, 64, 128, 256]):
        super().__init__(cfg, num_classes, in_channels, encoder, pretrained_path, 
                         audioset_pretrain, num_features)

        self.decoder = ConformerBlocks(encoder_dim=self.num_features[-1], num_layers=2)
    

class HTSAT(nn.Module):
    def __init__(self, cfg, num_classes, in_channels=7, audioset_pretrain=True,
                 pretrained_path='ckpts/HTSAT-fullset-imagenet-768d-32000hz.ckpt', 
                 **kwargs):
        super().__init__()
        
        data = cfg.data
        mel_bins = cfg.data.n_mels
        cfg_adapt = cfg.adapt
        self.label_res = 0.1
        self.num_classes = num_classes
        self.output_frames = None # int(data.train_chunklen_sec / 0.1)
        self.tgt_output_frames = int(10 / 0.1) # 10-second clip input to the model
        self.pred_res = int(data.sample_rate / data.hoplen * self.label_res)
        self.in_channels = in_channels
        
        # scalar
        self.scalar = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(in_channels)])
        
        # encoder
        self.encoder = HTSAT_Swin_Transformer(in_channels, mel_bins=mel_bins, 
                                              cfg_adapt=cfg_adapt, **kwargs)
        
        # fc
        self.tscam_conv = nn.Conv2d(
            in_channels = self.encoder.num_features,
            out_channels = self.num_classes * 3,
            kernel_size = (self.encoder.SF,3),
            padding = (0,1))

        self.fc = nn.Identity()
        self.final_act = nn.Tanh()

        if pretrained_path:
            print('Loading pretrained model from {}...'.format(pretrained_path))
            self.load_ckpts(pretrained_path, audioset_pretrain)
        
        self.freeze_layers_if_needed(cfg_adapt.get('method', ''))

        self.tscam_conv.requires_grad_(True)

    def freeze_layers_if_needed(self, adapt_method):
        if 'adapter' not in adapt_method:
            return
        
        ADAPTER = False
        self.requires_grad_(False)
        print('\n Freezing the model...')
        
        ''' Adapeter Tuning using monophonic/multichannel clips'''
        for name, param in self.named_parameters():
            if 'bias' in name: 
                param.requires_grad_(True)
                # ADAPTER = True
                print(f'param {name} is trainable')
            if 'adapter' in name or 'lora' in name:
                ADAPTER = True
                param.requires_grad_(True)
                print(f'param {name} is trainable')
        
        ''' Fine-tuning using monophonic clips'''
        if adapt_method == 'mono_adapter' and not ADAPTER:
            print('No adapter found, all parameters are trainable')
            self.requires_grad_(True)

    def load_ckpts(self, pretrained_path, audioset_pretrain=True):
        if audioset_pretrain:
            print('AudioSet-pretrained model...')
            htsat_ckpts = torch.load(pretrained_path, map_location='cpu')['state_dict']
            htsat_ckpts = {k.replace('sed_model.', ''): v for k, v in htsat_ckpts.items()}
            for key, value in self.encoder.state_dict().items():
                try:
                    if key == 'patch_embed.proj.weight':
                        paras = htsat_ckpts[key].repeat(1, self.in_channels, 1, 1) / self.in_channels
                        value.data.copy_(paras)
                    elif 'tscam_conv' not in key and 'head' not in key and 'adapter' not in key:
                        value.data.copy_(htsat_ckpts[key])
                    else: print(f'Skipping {key}...')
                except: print(key, value.shape, htsat_ckpts[key].shape)
            for ich in range(self.in_channels):
                self.scalar[ich].weight.data.copy_(htsat_ckpts['bn0.weight'])
                self.scalar[ich].bias.data.copy_(htsat_ckpts['bn0.bias'])
                self.scalar[ich].running_mean.copy_(htsat_ckpts['bn0.running_mean'])
                self.scalar[ich].running_var.copy_(htsat_ckpts['bn0.running_var'])
                self.scalar[ich].num_batches_tracked.copy_(htsat_ckpts['bn0.num_batches_tracked'])
        else:
            print('DataSynthSELD-pretrained model...')
            ckpt = torch.load(pretrained_path, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for idx, (key, value) in enumerate(self.state_dict().items()):
                if key.startswith(('fc.', 'head.', 'tscam_conv.')) or 'lora' in key or 'adapter' in key:
                    print(f'{idx+1}/{len(self.state_dict())}: Skipping {key}...')
                else:
                    try: value.data.copy_(ckpt[key])
                    except: print(f'{idx+1}/{len(self.state_dict())}: {key} not in ckpt.dict, skipping...')

    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, time_frames, mel_bins)
        """

        B, C, T, F = x.shape

        # Concatenate clips to a 10-second clip if necessary
        if self.output_frames is None:
            self.output_frames = int(T // self.pred_res)
        if self.output_frames < self.tgt_output_frames:
            assert self.output_frames == self.tgt_output_frames // 2, \
                'only support 5-second or 10-second clip or input to the model'
            factor = 2
            assert B % factor == 0, 'batch size should be a factor of {}'.format(factor)
            x = torch.cat((x[:B//factor, :, :-1], x[B//factor:, :, :-1]), dim=2)
        elif self.output_frames > self.tgt_output_frames:
            raise NotImplementedError('output_frames > tgt_output_frames is not implemented')

        # Compute scalar
        x = x.transpose(1, 3)
        for nch in range(x.shape[-1]):
            x[..., [nch]] = self.scalar[nch](x[..., [nch]])
        x = x.transpose(1, 3)

        x = self.encoder(x)
        x = self.tscam_conv(x)
        x = torch.flatten(x, 2) # (B, C, T)
        x = x.permute(0,2,1).contiguous() # B, T, C
        x = self.fc(x)

        # Match the output shape
        x = interpolate(x, ratio=self.encoder.time_res, method='bilinear')
        x = x[:, :self.output_frames * self.pred_res]
        if self.output_frames < self.tgt_output_frames:
            x = torch.cat((x[:, :self.output_frames], x[:, self.output_frames:]), dim=0)
        x = x.reshape(B, self.output_frames, self.pred_res, -1).mean(dim=2)

        x = self.final_act(x)

        return {
            'accdoa': x,
        }


class PASST(nn.Module):
    def __init__(self, cfg, num_classes, in_channels=7, pretrained_path=None,
                 audioset_pretrain=True, **kwargs):
        super().__init__()
        
        mel_bins = cfg.data.n_mels
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # scalar
        self.scalar = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(in_channels)])
        # encoder
        self.encoder = PaSST(in_channels, **kwargs)
        # fc
        self.fc = nn.Linear(self.encoder.num_features, num_classes * 3)
        self.final_act = nn.Tanh()

        if pretrained_path:
            print('Loading pretrained model from {}...'.format(pretrained_path))
            self.load_ckpts(pretrained_path, audioset_pretrain)

    def load_ckpts(self, pretrained_path, audioset_pretrain=True):
        if audioset_pretrain:
            passt_ckpt = torch.load(pretrained_path, map_location='cpu')
            for key, value in self.encoder.state_dict().items():
                if key == 'patch_embed.proj.weight':
                    paras = passt_ckpt[key].repeat(1, self.in_channels, 1, 1) / self.in_channels
                    value.data.copy_(paras)
                elif key == 'time_new_pos_embed':
                    time_new_pos_embed = passt_ckpt[key]
                    ori_time_len = time_new_pos_embed.shape[-1]
                    targ_time_len = self.encoder.time_new_pos_embed.shape[-1]
                    if ori_time_len >= targ_time_len:
                        start_index = int((ori_time_len - targ_time_len) / 2)
                        self.encoder.time_new_pos_embed.data.copy_(
                            time_new_pos_embed[:, :, :, start_index:start_index+targ_time_len])
                    else:
                        self.encoder.time_new_pos_embed.data.copy_(nn.functional.interpolate(
                            time_new_pos_embed, size=(1, targ_time_len), mode='bilinear'))
                elif key == 'freq_new_pos_embed':
                    freq_new_pos_embed = passt_ckpt[key]
                    ori_freq_len = freq_new_pos_embed.shape[-2]
                    targ_freq_len = self.encoder.freq_new_pos_embed.shape[-2]
                    if ori_freq_len >= targ_freq_len:
                        start_index = int((ori_freq_len - targ_freq_len) / 2)
                        self.encoder.freq_new_pos_embed.data.copy_(
                            freq_new_pos_embed[:, :, start_index:start_index+targ_freq_len, :])
                    else:
                        self.encoder.freq_new_pos_embed.data.copy_(nn.functional.interpolate(
                            freq_new_pos_embed, size=(1, targ_freq_len), mode='bilinear'))
                elif 'head' in key: 
                    if key in ['head.0.weight', 'head.0.bias']:
                        value.data.copy_(passt_ckpt[key])
                else:
                    value.data.copy_(passt_ckpt[key])
        else:
            ckpt = torch.load(pretrained_path, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for key, value in self.state_dict().items():
                if key.startswith('fc.'): print(f'Skipping {key}...')
                else: value.data.copy_(ckpt[key])
    
    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, time_frames, mel_bins)
        """

        # Compute scalar
        x = x.transpose(1, 3)
        for nch in range(x.shape[-1]):
            x[..., [nch]] = self.scalar[nch](x[..., [nch]])
        x = x.transpose(1, 3)

        x = self.encoder(x)[0]
        x = self.fc(x)
        x = self.final_act(x)

        return {
            'accdoa': x,
        }


        
    
