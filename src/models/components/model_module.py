import hydra
import numpy as np
import torch
import torch.optim as optim
import lightning as L
from itertools import combinations

from augment import *
from utils.config import get_afextractor
from utils.SELD_metrics import SELDMetrics
from torchmetrics import MeanMetric
from utils.utilities import get_pylogger
from utils.utilities import convert_ordinal
from utils.data_utilities import (accdoa_label_to_dcase_format, get_accdoa_labels, 
                                  multi_accdoa_to_dcase_format, get_multi_accdoa_labels,
                                  convert_output_format_cartesian_to_polar, 
                                  track_to_dcase_format, to_metrics_format)


class BaseModelModule(L.LightningModule):

    logging = get_pylogger(__name__)

    def __init__(self, cfg, dataset, valid_meta=None, test_meta=None):
        
        super().__init__()

        self.cfg = cfg
        self.num_classes = dataset.num_classes
        self.label_res = dataset.label_resolution
        self.method = self.cfg.model.method
        self.max_ov = dataset.max_ov
        self.stat = {'ov1': 0, 'ov2': 0, 'ov3': 0}
        
        self.net = None
        self.metrics = SELDMetrics(
            nb_classes=self.num_classes, 
            doa_threshold=20,)
        self.step_system_outputs = []
        self.num_preds_per_chunk = int(cfg.data.test_chunklen_sec / self.label_res)
        # self.valid_room = [_room+'_' for _room in cfg.data.valid_room]
        # self.valid_room.sort()

        self.get_num_frames = lambda x: int(
            np.ceil(x / self.num_preds_per_chunk) * self.num_preds_per_chunk)
        self.af_extractor = get_afextractor(cfg)
        self.configure_loss()
        if valid_meta is not None:
            self.valid_paths_dict, self.valid_gt_dcase_format = valid_meta
            self.paths_dict = self.valid_paths_dict
        if test_meta is not None:
            self.test_paths_dict = test_meta
            self.paths_dict = self.test_paths_dict

        self.train_loss_dict = torch.nn.ModuleDict()
        self.val_loss_dict = torch.nn.ModuleDict()
        for key in self.loss.loss_dict_keys:
            self.train_loss_dict.update({key: MeanMetric()})
            self.val_loss_dict.update({key: MeanMetric()})
        
        # Data Augmentations
        xy_ratio = cfg.data.sample_rate / cfg.data.hoplen * self.label_res
        self.data_aug = {
            'type': cfg.augment.type,
            'AugMix': cfg.augment.AugMix,
            'trackmix': hydra.utils.instantiate(cfg.augment.trackmix),
            'wavmix': hydra.utils.instantiate(cfg.augment.wavmix),
            'rotate': hydra.utils.instantiate(cfg.augment.rotate),
            'freqshift': hydra.utils.instantiate(cfg.augment.freqshift),
            'crop': hydra.utils.instantiate(cfg.augment.crop),
            'specaug': hydra.utils.instantiate(
                cfg.augment.specaug, xy_ratio=xy_ratio),
        }
        aug_TF = list(filter(lambda x: x not in ['rotate', 'wavmix'], 
                             self.data_aug['type']))
        self.aug_TF_comb = []
        for n in range(1, len(aug_TF) + 1):
            self.aug_TF_comb += combinations(aug_TF, n)

    def forward(self, x):
        return self.net(x)
    
    def data_copy(self, batch_x, batch_target):
        batch_x = torch.cat([batch_x] * 3, dim=0)
        for key, value in batch_target.items():
            if isinstance(value, torch.Tensor):
                batch_target[key] = torch.cat([value] * 3, dim=0)
            else:
                batch_target[key] = value * 3
        return batch_x, batch_target
    
    def augmix_data(self, batch_x, batch_target):

        N = len(batch_x) // 3
        batch_x_orig = batch_x[: N]
        batch_target_orig = {key: value[: N] for key, value in batch_target.items()}

        batch_x_1 = batch_x[N: 2 * N]
        batch_target_1 = {key: value[N: 2 * N] for key, value in batch_target.items()}

        batch_x_2= batch_x[2* N: 3 * N]
        batch_target_2 = {key: value[2* N: 3 * N] for key, value in batch_target.items()}

        batch_x_1, batch_target_1 = self.augment_data(batch_x_1, batch_target_1)
        batch_x_2, batch_target_2 = self.augment_data(batch_x_2, batch_target_2)
        batch_x = torch.cat((batch_x_orig, batch_x_1, batch_x_2), dim=0)
        for key in batch_target.keys():
            if 'label' in key:
                batch_target[key] = torch.cat(
                    [batch_target_orig[key], batch_target_1[key], batch_target_2[key]], dim=0)
            else:
                batch_target[key] = batch_target_orig[key] + batch_target_1[key] + batch_target_2[key]
        return batch_x, batch_target
    
    def augment_data(self, batch_x, batch_y=None):
        if self.data_aug['type'] and self.aug_TF_comb:
            aug_methods = list(random.choice(self.aug_TF_comb))
            random.shuffle(aug_methods)
            for aug_method in aug_methods:
                batch_x, batch_y = self.data_aug[aug_method](batch_x, batch_y)
        return batch_x, batch_y
    
    def standardize(self, batch_x):
        if self.af_extractor is not None:
            batch_x = self.af_extractor(batch_x)
        return batch_x
    
    def configure_optimizers(self):
        optimizer_params = self.cfg['model']['optimizer']
        lr_scheduler_params = self.cfg['model']['lr_scheduler']
        params = self.parameters()

        if self.cfg.model.optimizer.get('multi_opt', False):
            optim_params_1 = [param for name, param in self.named_parameters() 
                              if 'sed_encoder' not in name]
            optim_params_2 = [param for name, param in self.named_parameters() 
                              if 'sed_encoder' in name]
            print(len(optim_params_1), len(optim_params_2))
            params = [{'params': optim_params_2, **optimizer_params['kwargs1']},
                      {'params': optim_params_1}]
        
        optimizer = vars(optim)[optimizer_params['method']](
            params, **optimizer_params['kwargs'])
        lr_scheduler = vars(optim.lr_scheduler)[lr_scheduler_params['method']](
            optimizer, **lr_scheduler_params['kwargs'])
        return [optimizer], [lr_scheduler]
    
    def log_losses(self, losses_dict, set_type='train'):
        out_str = set_type + ": "
        for key, value in losses_dict.items():
            out_str += '{}: {:.4f}, '.format(key, value.compute())
            self.log(f'{set_type}/{key}', value.compute(), logger=True, 
                     on_epoch=True, on_step=False, sync_dist=True)
        self.logging.info(out_str)

    def log_metrics(self, values_dict, set_type='val/macro'):
        out_str = set_type + ": "
        for key, value in values_dict.items():
            if key in ['ER', 'SELD_scr']:
                out_str += '{}: {:.3f}, '.format(key, value)
            elif key in ['F', 'LR']:
                out_str += '{}: {:.1f}%, '.format(key, value*100)
            elif key in ['LE']:
                out_str += '{}: {:.1f}, '.format(key, value)
            else:
                out_str += '{}: {:.3f}, '.format(key, value)
            self.log(f'{set_type}/{key}', value, logger=True, 
                     on_epoch=True, on_step=False)
        self.logging.info(out_str)

    def configure_loss(self):
        self.loss = hydra.utils.instantiate(self.cfg.model.loss)
        for idx, loss_name in enumerate(self.loss.names):
            self.logging.info('{} is used as the {} loss.'.format(
                loss_name, convert_ordinal(idx + 1)))
    
    def pred_aggregation(self):
        outputs_gather = self.all_gather(self.step_system_outputs)
        self.step_system_outputs = []
        # gather outputs from all gpus
        if self.trainer.world_size > 1:
            outputs_gather = [{key: value.transpose(0,1).reshape(-1, *value.shape[2:]) 
                               for key, value in _output.items()} for _output in outputs_gather]
        
        sed_threshold = torch.tensor(self.cfg.sed_threshold)
        if self.method == 'accdoa':
            pred = [loc_pred['accdoa'].detach().cpu() for loc_pred in outputs_gather]
            pred = torch.cat(pred, dim=0)
            pred_sed, pred_doa = get_accdoa_labels(pred, self.num_classes, sed_threshold)
            pred_sed = pred_sed.numpy().reshape(-1, self.num_classes)
            pred_doa = pred_doa.float().numpy().reshape(-1, self.num_classes*3)
        
        elif self.method == 'einv2':
            pred_sed = [loc_pred['sed'].sigmoid_().detach().cpu() for loc_pred in outputs_gather]
            pred_doa = [loc_pred['doa'].detach().cpu() for loc_pred in outputs_gather]
            pred_sed = torch.cat(pred_sed, dim=0)
            sed_top_values, sed_top_indices = torch.topk(pred_sed, 1, dim=-1, largest=True)
            pred_sed = torch.zeros_like(pred_sed)
            pred_sed.scatter_(-1, sed_top_indices, sed_top_values)
            nb_batches, nb_frames, nb_tracks = pred_sed.shape[:3]
            # pred_sed = torch.where(pred_sed > self.cfg.sed_threshold, 1., 0.)
            pred_sed = pred_sed > sed_threshold
            pred_sed = pred_sed.numpy()
            pred_sed = pred_sed.reshape(nb_batches * nb_frames, nb_tracks, -1)
            pred_doa = torch.cat(pred_doa, dim=0).float().numpy()
            pred_doa = pred_doa.reshape(nb_batches * nb_frames, nb_tracks, -1)

        elif self.method == 'multi_accdoa':
            pred = [loc_pred['multi_accdoa'].detach().cpu() for loc_pred in outputs_gather]
            pred = torch.cat(pred, dim=0)
            if self.cfg.get('post_processing') == 'move_avg':
                print(pred.shape)
                pred = self.post_processing(preds=pred, method='move_avg', paths_dict=self.paths_dict)
            pred_sed, pred_doa = get_multi_accdoa_labels(pred, self.num_classes, sed_threshold)
            pred_sed = pred_sed.reshape(
                pred_sed.shape[0], pred_sed.shape[1]*pred_sed.shape[2], -1
                ).transpose(0, 1).numpy()
            pred_doa = pred_doa.reshape(
                pred_doa.shape[0], pred_doa.shape[1]*pred_doa.shape[2], -1
                ).transpose(0, 1).float().numpy()
        
        return pred_sed, pred_doa

    def convert_to_dcase_format_polar(self, pred_sed, pred_doa):
        if self.method == 'accdoa':
            pred_dcase_format = accdoa_label_to_dcase_format(
                pred_sed, pred_doa, self.num_classes)
            return convert_output_format_cartesian_to_polar(
                in_dict=pred_dcase_format)
        elif self.method == 'einv2':
            azi = np.arctan2(pred_doa[..., 1], pred_doa[..., 0])
            elev = np.arctan2(pred_doa[..., 2], np.sqrt(pred_doa[..., 0]**2 + pred_doa[..., 1]**2))
            pred_doa = np.stack((azi, elev), axis=-1) # (N, tracks, (azi, elev))
            return track_to_dcase_format(pred_sed, pred_doa)
        elif self.method == 'multi_accdoa':
            pred_sed, pred_doa = pred_sed.transpose(1, 0, 2), pred_doa.transpose(1, 0, 2)
            pred_dcase_format = multi_accdoa_to_dcase_format(
                pred_sed, pred_doa, nb_classes=self.num_classes)
            return convert_output_format_cartesian_to_polar(
                in_dict=pred_dcase_format)
    
    def update_metrics(self, pred_dcase_format, gt_dcase_format, 
                       num_frames, metrics_all: SELDMetrics = None,
                       metric_room_wise:SELDMetrics = None):
        #### compute metrics of each clip ####
        pred_metrics_format = to_metrics_format(
            label_dict=pred_dcase_format, num_frames=num_frames)
        gt_metrics_format = to_metrics_format(
            label_dict=gt_dcase_format, num_frames=num_frames)
        if metrics_all is None:
            metrics_all = self.metrics
        metrics_all.update_seld_scores(
            pred=pred_metrics_format, gt=gt_metrics_format)
        # update room_wise metrics while meta-learning
        if metric_room_wise is not None:
            metric_room_wise.update_seld_scores(
                pred=pred_metrics_format, gt=gt_metrics_format)

    
    def training_step(self, batch_sample, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch_sample, batch_idx):
        raise NotImplementedError

    def test_step(self, batch_sample, batch_idx):
        raise NotImplementedError
    
    def post_processing(self, batch_sample=None, preds=None, paths_dict={},
                        method='ACS', output_format='multi_accdoa'):
        outputs = []
        if method == 'ACS':
            trans_dict = {(0,1,2): (1,2,3), (1,0,2): (3,2,1)}
            signs = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1],
                     [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]
            for sign in signs:
                for trans_y, trans_x in trans_dict.items():
                    sign_x, sign_y, sign_z = sign
                    xx, yy, zz = trans_y
                    s_x, s_y, s_z = trans_x
                    x = torch.stack((batch_sample[:, 0], sign_y*batch_sample[:, s_x], 
                                     sign_z*batch_sample[:, s_y], sign_x*batch_sample[:, s_z]), axis=1)
                    x = self.standardize(x)
                    y = self.forward(x)
                    if output_format == 'multi_accdoa':
                        y = y['multi_accdoa']
                        B, T = y.shape[:2]
                        y = y.reshape(B, T, 3, 3, -1)
                        y = torch.stack((sign_x*y[..., 0, :], sign_y*y[..., 1, :], sign_z*y[..., 2, :]), dim=-2)
                        y = torch.stack((y[..., xx, :], y[..., yy, :], y[..., zz, :]), dim=-2)
                        y = y.reshape(B, T, -1)
                    elif output_format == 'accdoa':
                        y = y['accdoa']
                        B, T = y.shape[:2]
                        y = y.reshape(B, T, 3, -1)
                        y = torch.stack((sign_x*y[..., 0, :], sign_y*y[..., 1, :], sign_z*y[..., 2, :]), dim=-2)
                        y = torch.stack((y[..., xx, :], y[..., yy, :], y[..., zz, :]), dim=-2)
                        y = y.reshape(B, T, -1)
                    else:
                        raise NotImplementedError
                    outputs.append(y)
            if output_format != 'einv2':
                return {output_format: torch.mean(torch.stack(outputs), dim=0)}
        elif method == 'move_avg':
            batch_ind = 0
            seg_lens = paths_dict.values()
            test_chunklen_sec = self.cfg.data.test_chunklen_sec
            test_hoplen_sec = self.cfg.data.test_hoplen_sec
            chunk_len = int(test_hoplen_sec / self.label_res)
            assert test_chunklen_sec % test_hoplen_sec == 0
            for seg_len in seg_lens:
                num_chunks = np.ceil((seg_len - test_chunklen_sec / self.label_res) / chunk_len).astype(int) + 1
                valid_num_chunks = np.ceil(seg_len / chunk_len).astype(int)
                tgt_seg_len = self.get_num_frames(seg_len)
                res = {i: [] for i in range(valid_num_chunks)}
                local_preds = preds[batch_ind:batch_ind+num_chunks]
                for i in range(valid_num_chunks):
                    left_i = int(max(0, i - test_chunklen_sec // test_hoplen_sec + 1))
                    right_i =int(min(i + 1, num_chunks))
                    for j in range(left_i, right_i):
                        res[i].append(local_preds[j,(i-j)*chunk_len:(i-j+1)*chunk_len])
                res = {key: torch.stack(value, dim=0).mean(dim=0) for key, value in res.items()}
                res = torch.cat([res[k] for k in range(valid_num_chunks)], dim=0)
                if res.shape[0] < tgt_seg_len:
                    res = torch.cat([res, torch.zeros(tgt_seg_len - res.shape[0], *res.shape[1:])], dim=0)
                else:
                    res = res[:tgt_seg_len]
                outputs.append(res)
                batch_ind += num_chunks
            return torch.cat(outputs).unsqueeze(0)