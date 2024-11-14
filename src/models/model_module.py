from pathlib import Path
import logging

import models
from models.components.model_module import BaseModelModule
from utils.data_utilities import write_output_format_file

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

ModelMoodule = {
    'accdoa': models.accdoa,
    'einv2': models.einv2,
    'multi_accdoa': models.multi_accdoa,
}


class SELDModelModule(BaseModelModule):
    
    def setup(self, stage):

        audio_feature = self.cfg.data.audio_feature
        kwargs = self.cfg.model.kwargs
        if audio_feature in ['logmelIV', 'salsa', 'salsalite']:
            in_channels = 7
        elif audio_feature in ['logmelgcc']:
            in_channels = 10
        elif audio_feature in ['logmel']:
            in_channels = 1
        
        self.net = vars(ModelMoodule[self.method])[self.cfg.model.backbone](
            self.cfg, self.num_classes, in_channels, **kwargs)
        if self.cfg.compile:
            self.logging.info('Compiling model')
            self.net = torch.compile(self.net)

        self.logging.info("Number of parameters of net: " + 
                            f"{sum(p.numel() for p in self.net.parameters())}")
        if stage == 'test':
            logger = logging.getLogger()
            log_filename = logger.handlers[1].baseFilename
            self.submissions_dir = Path(log_filename).parent / 'submissions'
            self.submissions_dir.mkdir(exist_ok=True)
    
    def common_step(self, batch_x, batch_y=None):

        # Augment before feature extraction
        if self.training:
            if self.data_aug['AugMix']: 
                batch_x, batch_y = self.data_copy(batch_x, batch_y) 
            if 'rotate' in self.data_aug['type']:
                batch_x, batch_y = self.data_aug['rotate'](batch_x, batch_y)
            if 'wavmix' in self.data_aug['type']:
                batch_x, batch_y = self.data_aug['wavmix'](batch_x, batch_y)
        
        batch_x = self.standardize(batch_x)
        
        # Augment after feature extraction
        if self.training:
            if self.data_aug['AugMix']:
                batch_x, batch_y = self.augmix_data(batch_x, batch_y)
            else:
                batch_x, batch_y = self.augment_data(batch_x, batch_y)
        batch_x = self.forward(batch_x)

        return batch_x, batch_y

    def training_step(self, batch_sample, batch_idx):
        self.stat['ov1'] += np.sum(np.array(batch_sample['ov']) == '1')
        self.stat['ov2'] += np.sum(np.array(batch_sample['ov']) == '2')
        self.stat['ov3'] += np.sum(np.array(batch_sample['ov']) == '3')
        batch_data = batch_sample['data']
        batch_target = {key: value for key, value in batch_sample.items() if 'data' not in key}
        batch_pred, batch_target = self.common_step(batch_data, batch_target)
        loss_dict = self.loss(batch_pred, batch_target)
        loss_dict[self.loss.loss_type] = loss_dict[self.loss.loss_type]
        for key in loss_dict.keys():
            self.train_loss_dict[key].update(loss_dict[key])
        return loss_dict[self.loss.loss_type] 

    def validation_step(self, batch_sample, batch_idx):
        batch_data = batch_sample['data']
        batch_target = {key: value for key, value in batch_sample.items()
                        if 'label' in key}
        if self.cfg.get('post_processing') == 'ACS':
            batch_pred = self.post_processing(batch_data, method='ACS', output_format=self.method)
        else:
            batch_pred = self.common_step(batch_data)[0]
        self.step_system_outputs.append(batch_pred)
        loss_dict = self.loss(batch_pred, batch_target)
        for key in loss_dict.keys():
            self.val_loss_dict[key].update(loss_dict[key])

    def on_validation_epoch_start(self):
        print(self.stat)
        self.stat = {'ov1': 0, 'ov2': 0, 'ov3': 0}
        self.metrics.reset()
    
    def on_load_checkpoint(self, checkpoint):
        if self.cfg.compile:
            return
        keys_list = list(checkpoint['state_dict'].keys())
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                checkpoint['state_dict'][deal_key] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]

    def on_validation_epoch_end(self):
        pred_sed, pred_doa = self.pred_aggregation()
        
        ######## compute metrics ########
        frame_ind = 0
        paths = tqdm(self.valid_paths_dict.keys(), 
                     desc='Computing metrics for validation set')
        f = open('metrics.csv', 'w')
        for path in paths:
            loc_frames = self.valid_paths_dict[path]
            num_frames = self.get_num_frames(loc_frames)
            pred_dcase_format = self.convert_to_dcase_format_polar(
                pred_sed=pred_sed[frame_ind:frame_ind+loc_frames],
                pred_doa=pred_doa[frame_ind:frame_ind+loc_frames])
            gt_dcase_format = self.valid_gt_dcase_format[path]
            # if '&' in str(path):
            self.update_metrics(
                pred_dcase_format=pred_dcase_format, 
                gt_dcase_format=gt_dcase_format, 
                num_frames=loc_frames)
            frame_ind += num_frames
            # res = self.metrics.compute_seld_scores()[0]
            # print(path, res['LE'], res['LR'])
            # self.metrics.reset()
            # f.write(f"{path},{res['LE']},{res['LR']}\n")
        f.close()

        ######## logging ########
        self.logging.info("-------------------------------------------"
                 + "---------------------------------------")
        metric_dict, _ = self.metrics.compute_seld_scores(average='macro')
        self.log_metrics(metric_dict, set_type='val/macro')
        metric_dict, _ = self.metrics.compute_seld_scores(average='micro')
        self.log_metrics(metric_dict, set_type='val/micro')
        self.log_losses(self.val_loss_dict, set_type='val')

    
    def on_train_epoch_end(self):
        lr = self.optimizers().param_groups[0]['lr']
        max_epochs = self.cfg.trainer.max_epochs
        self.log_losses(self.train_loss_dict, set_type='train')
        self.log('lr', lr)
        self.logging.info(f"Epoch/Total Epoch: {self.current_epoch+1}/{max_epochs}, LR: {lr}")
        self.logging.info("-------------------------------------------"
                 + "---------------------------------------")

    def on_test_epoch_start(self):
        self.step_system_outputs = []
    
    def test_step(self, batch_sample, batch_idx):
        batch_data = batch_sample['data']
        batch_pred = self.common_step(batch_data)[0]
        self.step_system_outputs.append(batch_pred)
    
    def on_test_epoch_end(self):
        pred_sed, pred_doa = self.pred_aggregation()
        
        frame_ind = 0
        for path in self.test_paths_dict.keys():
            loc_frames = self.test_paths_dict[path]
            # fn = path.split('/')[-1].replace('h5','csv')
            fn = Path(path).stem + '.csv'
            num_frames = self.get_num_frames(loc_frames)
            pred_dcase_format = self.convert_to_dcase_format_polar(
                pred_sed=pred_sed[frame_ind:frame_ind+loc_frames],
                pred_doa=pred_doa[frame_ind:frame_ind+loc_frames])
            csv_path = self.submissions_dir.joinpath(fn)
            write_output_format_file(csv_path, pred_dcase_format)
            frame_ind += num_frames
        self.logging.info('Rsults are saved to {}\n'.format(str(self.submissions_dir)))