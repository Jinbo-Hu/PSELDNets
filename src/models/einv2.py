from copy import deepcopy

import torch
import torch.nn as nn
from models.components.conformer import ConformerBlocks
from models.components.model_utilities import (Decoder, CrossStitch, 
                                               get_linear_layer, get_conv2d_layer)
from models.components.backbone import CNN8, CNN12
from models.components.htsat import HTSAT_Swin_Transformer
from models.components.passt import PaSST
from models.components.utils import interpolate
from utils.utilities import get_pylogger

log = get_pylogger(__name__)


class CRNN(nn.Module):
    def __init__(self, cfg, num_classes, in_channels=7, encoder='CNN8', pretrained_path=None,
                 audioset_pretrain=True, num_features=[32, 64, 128, 256]):
        super().__init__()

        self.sed_in_channels = 4
        self.doa_in_channels = in_channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        data = cfg.data
        mel_bins = cfg.data.n_mels
        self.label_res = 0.1
        self.interpolate_time_ratio = 2 ** 3
        self.output_frames = None #int(data.train_chunklen_sec / 0.1)
        self.pred_res = int(data.sample_rate / data.hoplen * self.label_res) # 10

        self.num_features = num_features
        self.scalar = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(in_channels)])
        self.stitch = nn.ModuleList([CrossStitch(num_feature) for num_feature in num_features])
        self.stitch.append(CrossStitch(num_features[-1]))
        self.stitch.append(CrossStitch(num_features[-1]))
        
        if encoder == 'CNN8':
            self.sed_convs = CNN8(self.sed_in_channels, num_features)
            self.doa_convs = CNN8(self.doa_in_channels, num_features)
        elif encoder == 'CNN12':
            self.sed_convs = CNN12(self.sed_in_channels, num_features)
            self.doa_convs = CNN12(self.doa_in_channels, num_features)
            if pretrained_path:
                log.info('\n Loading pretrained model from {}... \n'.format(pretrained_path))
                self.load_ckpts(pretrained_path, audioset_pretrain)
        else:
            raise NotImplementedError(f'encoder {encoder} is not implemented')
        
        num_layers = cfg.model.num_decoder_layers
        self.sed_track1 = Decoder(cfg.model.decoder, num_features[-1], num_layers)
        self.sed_track2 = Decoder(cfg.model.decoder, num_features[-1], num_layers)
        self.sed_track3 = Decoder(cfg.model.decoder, num_features[-1], num_layers)
        self.doa_track1 = Decoder(cfg.model.decoder, num_features[-1], num_layers)
        self.doa_track2 = Decoder(cfg.model.decoder, num_features[-1], num_layers)
        self.doa_track3 = Decoder(cfg.model.decoder, num_features[-1], num_layers)
        
        self.fc_sed_track1 = nn.Linear(num_features[-1], num_classes, bias=True, )
        self.fc_sed_track2 = nn.Linear(num_features[-1], num_classes, bias=True, )
        self.fc_sed_track3 = nn.Linear(num_features[-1], num_classes, bias=True, )
        self.fc_doa_track1 = nn.Linear(num_features[-1], 3, bias=True, )
        self.fc_doa_track2 = nn.Linear(num_features[-1], 3, bias=True, )
        self.fc_doa_track3 = nn.Linear(num_features[-1], 3, bias=True, )
        self.final_act_sed = nn.Sequential() # nn.Sigmoid()
        self.final_act_doa = nn.Tanh()
    
    def load_ckpts(self, pretrained_path, audioset_pretrain=True):
        if audioset_pretrain:
            CNN14_ckpt = torch.load(pretrained_path, map_location='cpu')['model']
            for key, value in self.sed_convs.state_dict().items():
                if key == 'conv_block1.conv1.weight':
                    paras = CNN14_ckpt[key].repeat(1, self.sed_in_channels, 1, 1) / self.sed_in_channels
                    value.data.copy_(paras)
                else: value.data.copy_(CNN14_ckpt[key])
            for key, value in self.doa_convs.state_dict().items():
                if key == 'conv_block1.conv1.weight':
                    paras = CNN14_ckpt[key].repeat(1, self.doa_in_channels, 1, 1) / self.doa_in_channels
                    value.data.copy_(paras)
                else: value.data.copy_(CNN14_ckpt[key])
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
                if key.startswith('fc_sed'):
                    log.info(f'Skipping {key}...')
                else:
                    value.data.copy_(ckpt[key])

    def forward(self, x,):
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
        
        x_sed = x[:, :self.sed_in_channels]
        x_doa = x

        # encoder
        for idx, (conv_sed, conv_doa) in enumerate(zip(self.sed_convs.convs[:-1], 
                                                       self.doa_convs.convs[:-1])):
            x_sed = conv_sed(x_sed)
            x_doa = conv_doa(x_doa)
            x_sed, x_doa = self.stitch[idx](x_sed, x_doa)
        x_sed = self.sed_convs.convs[-1](x_sed)
        x_doa = self.doa_convs.convs[-1](x_doa)
        x_sed = x_sed.mean(dim=3) # (N, C, T)
        x_doa = x_doa.mean(dim=3) # (N, C, T)

        # decoder
        x_sed = x_sed.permute(0, 2, 1) # (N, T, C)
        x_doa = x_doa.permute(0, 2, 1) # (N, T, C)

        x_sed_1 = self.sed_track1(x_sed) # (N, T, C)
        x_doa_1 = self.doa_track1(x_doa) # (N, T, C)
        x_sed_2 = self.sed_track2(x_sed) # (N, T, C)   
        x_doa_2 = self.doa_track2(x_doa) # (N, T, C)
        x_sed_3 = self.sed_track3(x_sed) # (N, T, C)
        x_doa_3 = self.doa_track3(x_doa) # (N, T, C)

        x_sed_1, x_doa_1 = self.stitch[-3](x_sed_1, x_doa_1)
        x_sed_2, x_doa_2 = self.stitch[-2](x_sed_2, x_doa_2)
        x_sed_3, x_doa_3 = self.stitch[-1](x_sed_3, x_doa_3)

        # upsample
        target_shape = (N, self.output_frames, self.pred_res, -1)
        x_sed_1 = interpolate(x_sed_1, ratio=self.interpolate_time_ratio)
        x_sed_2 = interpolate(x_sed_2, ratio=self.interpolate_time_ratio)
        x_sed_3 = interpolate(x_sed_3, ratio=self.interpolate_time_ratio)
        x_doa_1 = interpolate(x_doa_1, ratio=self.interpolate_time_ratio)
        x_doa_2 = interpolate(x_doa_2, ratio=self.interpolate_time_ratio)
        x_doa_3 = interpolate(x_doa_3, ratio=self.interpolate_time_ratio)
        x_sed_1 = x_sed_1.reshape(*target_shape).mean(dim=2)
        x_sed_2 = x_sed_2.reshape(*target_shape).mean(dim=2)
        x_sed_3 = x_sed_3.reshape(*target_shape).mean(dim=2)
        x_doa_1 = x_doa_1.reshape(*target_shape).mean(dim=2)
        x_doa_2 = x_doa_2.reshape(*target_shape).mean(dim=2)
        x_doa_3 = x_doa_3.reshape(*target_shape).mean(dim=2)

        # fc
        x_sed_1 = self.final_act_sed(self.fc_sed_track1(x_sed_1))
        x_sed_2 = self.final_act_sed(self.fc_sed_track2(x_sed_2))
        x_sed_3 = self.final_act_sed(self.fc_sed_track3(x_sed_3))
        x_sed = torch.stack((x_sed_1, x_sed_2, x_sed_3), 2)
        x_doa_1 = self.final_act_doa(self.fc_doa_track1(x_doa_1))
        x_doa_2 = self.final_act_doa(self.fc_doa_track2(x_doa_2))
        x_doa_3 = self.final_act_doa(self.fc_doa_track3(x_doa_3))
        x_doa = torch.stack((x_doa_1, x_doa_2, x_doa_3), 2)

        output = {
            'sed': x_sed,
            'doa': x_doa,
        }

        return output


class ConvConformer(CRNN):
    def __init__(self, num_classes, in_channels=7, encoder='CNN8', pretrained_path=None,
                 mel_bins=64, num_features=[32, 64, 128, 256], cfg=None):
        super().__init__(num_classes, in_channels, encoder, pretrained_path, mel_bins, num_features)
        
        num_layers = cfg.model.num_decoder_layers
        self.sed_track1 = ConformerBlocks(encoder_dim=self.num_features[-1], num_layers=num_layers)
        self.sed_track2 = ConformerBlocks(encoder_dim=self.num_features[-1], num_layers=num_layers)
        self.sed_track3 = ConformerBlocks(encoder_dim=self.num_features[-1], num_layers=num_layers)

        self.doa_track1 = ConformerBlocks(encoder_dim=self.num_features[-1], num_layers=num_layers)
        self.doa_track2 = ConformerBlocks(encoder_dim=self.num_features[-1], num_layers=num_layers)
        self.doa_track3 = ConformerBlocks(encoder_dim=self.num_features[-1], num_layers=num_layers)


class HTSAT(nn.Module):
    def __init__(self, cfg, num_classes, in_channels=7, audioset_pretrain=True,
                 pretrained_path='ckpts/HTSAT-fullset-imagenet-768d-32000hz.ckpt', 
                 **kwargs):
        super().__init__()
        
        data = cfg.data
        mel_bins = cfg.data.n_mels
        self.label_res = 0.1
        self.num_classes = num_classes
        self.output_frames = None #int(data.train_chunklen_sec / 0.1)
        self.tgt_output_frames = int(10 / 0.1) # 10-second clip input to the model
        self.pred_res = int(data.sample_rate / data.hoplen * self.label_res)
        self.sed_in_channels = 4
        self.doa_in_channels = in_channels
        self.in_channels = in_channels
        
        # scalar
        self.scalar = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(in_channels)])
        
        # encoder
        self.sed_encoder = HTSAT_Swin_Transformer(self.sed_in_channels, mel_bins=mel_bins, **kwargs)
        self.doa_encoder = HTSAT_Swin_Transformer(self.doa_in_channels, mel_bins=mel_bins, **kwargs)
        
        # soft-parameter sharing
        num_feats = [kwargs['embed_dim'] * (2 ** i_layer) for i_layer in range(len(kwargs['depths']))]
        self.stitch1 = nn.ModuleList([CrossStitch(num_feat) for num_feat in num_feats])
        
        self.sed_tscam_conv = nn.Conv2d(
            in_channels = num_feats[-1],
            out_channels = self.num_classes * 3,
            kernel_size = (self.sed_encoder.SF,3),
            padding = (0,1))
        self.doa_tscam_conv = nn.Conv2d(
            in_channels = num_feats[-1],
            out_channels = 3 * 3,
            kernel_size = (self.doa_encoder.SF,3),
            padding = (0,1))
        
        self.final_act_sed = nn.Identity()
        self.final_act_doa = nn.Tanh()

        if pretrained_path:
            log.info('\n Loading pretrained model from {}... \n'.format(pretrained_path))
            self.load_ckpts(pretrained_path, audioset_pretrain)
        
        log.info('Scalar and stitch are trainable...')
        log.info(f'Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')
        log.info(f'Non-trainable parameters: {sum(p.numel() for p in self.parameters() if not p.requires_grad)}')

    def load_ckpts(self, pretrained_path, audioset_pretrain=True):
        if audioset_pretrain:
            htsat_ckpts = torch.load(pretrained_path, map_location='cpu')['state_dict']
            htsat_ckpts = {k.replace('sed_model.', ''): v for k, v in htsat_ckpts.items()}
            
            log.info('\n Loading weights for sed_encoder... \n')
            for key, value in self.sed_encoder.state_dict().items():
                if key == 'patch_embed.proj.weight':
                    paras = htsat_ckpts[key].repeat(1, self.sed_in_channels, 1, 1) / self.sed_in_channels
                    value.data.copy_(paras)
                else: value.data.copy_(htsat_ckpts[key])

            log.info('\n Loading weights for doa_encoder... \n')
            for key, value in self.doa_encoder.state_dict().items():
                if key == 'patch_embed.proj.weight':
                    paras = htsat_ckpts[key].repeat(1, self.doa_in_channels, 1, 1) / self.doa_in_channels
                    value.data.copy_(paras)
                else: value.data.copy_(htsat_ckpts[key])

            for ich in range(self.in_channels):
                self.scalar[ich].weight.data.copy_(htsat_ckpts['bn0.weight'])
                self.scalar[ich].bias.data.copy_(htsat_ckpts['bn0.bias'])
                self.scalar[ich].running_mean.copy_(htsat_ckpts['bn0.running_mean'])
                self.scalar[ich].running_var.copy_(htsat_ckpts['bn0.running_var'])
                self.scalar[ich].num_batches_tracked.copy_(htsat_ckpts['bn0.num_batches_tracked'])
        else:
            ckpt = torch.load(pretrained_path, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            state_dict = self.state_dict()
            for key, value in ckpt.items():
                if key.startswith(('sed_tscam_conv.', 'head', 'af_extractor')):
                    log.info(f'Skipping {key}...')
                else: state_dict[key].data.copy_(value)

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
                'only support 5-second or 10-second clip to be input to the model'
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

        # Rewrite the forward function of the encoders
        x_sed = self.sed_encoder.forward_patch(x[:, :self.sed_in_channels])
        x_doa = self.doa_encoder.forward_patch(x)
        for sed_layer, doa_layer, stitch in zip(self.sed_encoder.layers, 
                                                self.doa_encoder.layers, 
                                                self.stitch1):
            x_sed, x_doa = stitch(x_sed, x_doa)
            x_sed = sed_layer(x_sed)[0]
            x_doa = doa_layer(x_doa)[0]
        x_sed = self.sed_encoder.forward_reshape(x_sed)
        x_doa = self.doa_encoder.forward_reshape(x_doa)

        pred_sed = torch.flatten(self.sed_tscam_conv(x_sed), 2).permute(0,2,1).contiguous()
        pred_doa = torch.flatten(self.doa_tscam_conv(x_doa), 2).permute(0,2,1).contiguous()
        pred_sed = interpolate(pred_sed, ratio=self.sed_encoder.time_res, method='bilinear')[:,:self.tgt_output_frames*self.pred_res]
        pred_doa = interpolate(pred_doa, ratio=self.doa_encoder.time_res, method='bilinear')[:,:self.tgt_output_frames*self.pred_res]
        if self.output_frames < self.tgt_output_frames:
            x_output_frames = self.output_frames * self.pred_res
            pred_sed = torch.cat((pred_sed[:, :x_output_frames], pred_sed[:, x_output_frames:]), dim=0)
            pred_doa = torch.cat((pred_doa[:, :x_output_frames], pred_doa[:, x_output_frames:]), dim=0)
        pred_sed = pred_sed.reshape(B, self.output_frames, self.pred_res, 3, -1).mean(dim=2)
        pred_doa = pred_doa.reshape(B, self.output_frames, self.pred_res, 3, -1).mean(dim=2)

        pred_sed = self.final_act_sed(pred_sed)
        pred_doa = self.final_act_doa(pred_doa)

        return {
            'sed': pred_sed,
            'doa': pred_doa,
        }


class PASST(nn.Module):
    def __init__(self, cfg, num_classes, in_channels=7, pretrained_path=None,
                 audioset_pretrain=True, **kwargs):
        super().__init__()
        
        mel_bins = cfg.data.n_mels
        self.num_classes = num_classes
        self.sed_in_channels = 4
        self.doa_in_channels = in_channels
        self.in_channels = in_channels
        
        # scalar
        self.scalar = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(in_channels)])
        
        # encoder
        self.sed_encoder = PaSST(self.sed_in_channels, **kwargs)
        self.doa_encoder = PaSST(self.doa_in_channels, **kwargs)
        
        # soft-parameter sharing
        self.ps_gap = cfg.model.ps_gap
        num_stitch = (kwargs['depth'] - 1) // self.ps_gap + 1
        self.stitch1 = nn.ModuleList([CrossStitch(kwargs['embed_dim']) for _ in range(num_stitch)])

        # decoder
        num_layers = cfg.model.num_decoder_layers
        self.sed_decoder = nn.ModuleList([Decoder(
            cfg.model.decoder, kwargs['embed_dim'], num_layers=num_layers) for _ in range(3)])
        self.doa_decoder = nn.ModuleList([Decoder(
            cfg.model.decoder, kwargs['embed_dim'], num_layers=num_layers) for _ in range(3)])
        self.stitch2 = nn.ModuleList([CrossStitch(kwargs['embed_dim']) for _ in range(3)])

        self.fc_sed = nn.ModuleList([nn.Linear(kwargs['embed_dim'], num_classes, bias=True, ) for _ in range(3)])
        self.fc_doa = nn.ModuleList([nn.Linear(kwargs['embed_dim'], 3, bias=True, ) for _ in range(3)])
        
        self.final_act_sed = nn.Sequential() # nn.Sigmoid()
        self.final_act_doa = nn.Tanh()

        if pretrained_path:
            log.info('\n Loading pretrained model from {}... \n'.format(pretrained_path))
            self.load_ckpts(pretrained_path, audioset_pretrain)

    def load_ckpts(self, pretrained_path, audioset_pretrain=True):

        def load_ckpt(ckpt, model, in_channels):
            for key, value in model.state_dict().items():
                if key == 'patch_embed.proj.weight':
                    paras = ckpt[key].repeat(1, in_channels, 1, 1) / in_channels
                    value.data.copy_(paras)
                elif key == 'time_new_pos_embed':
                    time_new_pos_embed = ckpt[key]
                    ori_time_len = time_new_pos_embed.shape[-1]
                    targ_time_len = model.time_new_pos_embed.shape[-1]
                    if ori_time_len >= targ_time_len:
                        start_index = int((ori_time_len - targ_time_len) / 2)
                        model.time_new_pos_embed.data.copy_(
                            time_new_pos_embed[:, :, :, start_index:start_index+targ_time_len])
                    else:
                        model.time_new_pos_embed.data.copy_(nn.functional.interpolate(
                            time_new_pos_embed, size=(1, targ_time_len), mode='bilinear'))
                elif key == 'freq_new_pos_embed':
                    freq_new_pos_embed = ckpt[key]
                    ori_freq_len = freq_new_pos_embed.shape[-2]
                    targ_freq_len = model.freq_new_pos_embed.shape[-2]
                    if ori_freq_len >= targ_freq_len:
                        start_index = int((ori_freq_len - targ_freq_len) / 2)
                        model.freq_new_pos_embed.data.copy_(
                            freq_new_pos_embed[:, :, start_index:start_index+targ_freq_len, :])
                    else:
                        model.freq_new_pos_embed.data.copy_(nn.functional.interpolate(
                            freq_new_pos_embed, size=(1, targ_freq_len), mode='bilinear'))
                elif 'head' in key: 
                    if key in ['head.0.weight', 'head.0.bias']:
                        value.data.copy_(ckpt[key])
                else:
                    value.data.copy_(ckpt[key])

        if audioset_pretrain:
            passt_ckpt = torch.load(pretrained_path, map_location='cpu')
            load_ckpt(passt_ckpt, self.sed_encoder, self.sed_in_channels)
            load_ckpt(passt_ckpt, self.doa_encoder, self.doa_in_channels)
        else:
            ckpt = torch.load(pretrained_path, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for key, value in self.state_dict().items():
                if key.startswith('fc_sed'):
                    log.info(f'Skipping {key}...')
                else:
                    value.data.copy_(ckpt[key])
    
    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, time_frames, mel_bins)
        """

        # Compute scalar
        x = x.transpose(1, 3)
        for nch in range(x.shape[-1]):
            x[..., [nch]] = self.scalar[nch](x[..., [nch]])
        x = x.transpose(1, 3)

        x_sed = self.sed_encoder.forward_before(x[:, :self.sed_in_channels])
        x_doa = self.doa_encoder.forward_before(x)
        for layer_idx, (sed_layer, doa_layer) in enumerate(
            zip(self.sed_encoder.blocks, self.doa_encoder.blocks)):
            if layer_idx % self.ps_gap == 0:
                x_sed, x_doa = self.stitch1[layer_idx // self.ps_gap](x_sed, x_doa)
            x_sed = sed_layer(x_sed)
            x_doa = doa_layer(x_doa)
        x_sed = self.sed_encoder.forward_after(x_sed)[0]
        x_doa = self.doa_encoder.forward_after(x_doa)[0]

        preds_sed, preds_doa = [], []
        for decoder_sed, decoder_doa, fc_sed, fc_doa, stitch in zip(
            self.sed_decoder, self.doa_decoder, self.fc_sed, self.fc_doa, self.stitch2
            ):
            pred_sed = decoder_sed(x_sed)
            pred_doa = decoder_doa(x_doa)
            pred_sed, pred_doa = stitch(pred_sed, pred_doa)
            preds_sed.append(fc_sed(pred_sed))
            preds_doa.append(fc_doa(pred_doa))            

        preds_sed = torch.stack(preds_sed, 2)
        preds_doa = torch.stack(preds_doa, 2)
        preds_sed = self.final_act_sed(preds_sed)
        preds_doa = self.final_act_doa(preds_doa)

        return {
            'sed': preds_sed,
            'doa': preds_doa
        }