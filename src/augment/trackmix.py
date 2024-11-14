import numpy as np
import torch
from torch.distributions.beta import Beta


class TrackMix:
    def __init__(self, alpha=0.5):
        """ Mixup for data augmentation
        Args:
            alpha: 0.0-1.0, 0.5
        """
        self.beta = Beta(alpha, alpha)
    
    def __call__(self, batch_x, batch_target):

        ov = batch_target['ov']
        idx_ov1 = [n for n in range(len(ov)) if ov[n] == '1']
        new_idx_ov1 = np.random.permutation(idx_ov1)

        N = len(idx_ov1)
        if N == 0:
            return batch_x, batch_target
        
        label_keys = [k for k in batch_target.keys() if 'label' in k]
        label_key = label_keys[0]
        lams = self.beta.sample((N,)).to(batch_x.device)
        x_shape = (N, ) + (1,) * (batch_x.ndim - 1)
        lams_x = lams.reshape(x_shape)
        if len(label_keys) == 2: # sed_label and doa_label
            y_sed_shape = (N, ) + (1,) * (batch_target['sed_label'].ndim - 2)
            y_doa_shape = (N, ) + (1,) * (batch_target['doa_label'].ndim - 2)
            lams_y_sed = lams.reshape(y_sed_shape)
            lams_y_doa = lams.reshape(y_doa_shape)
        else: # accdoa_label or adpit_label
            y_shape = (N, ) + (1,) * (batch_target[label_key].ndim - 1)
            lams_y = lams.reshape(y_shape)
        
        batch_x[idx_ov1] = lams_x * batch_x[idx_ov1] + (1. - lams_x) * batch_x[new_idx_ov1]
        if len(label_keys) == 2: # sed_label and doa_label
            batch_target['sed_label'][idx_ov1] = torch.stack((
                lams_y_sed * batch_target['sed_label'][idx_ov1][:, :, 0],
                (1. - lams_y_sed) * batch_target['sed_label'][new_idx_ov1][:, :, 0], 
                torch.zeros_like(batch_target['sed_label'][idx_ov1][:, :, 0])
                ), dim=2)
            # batch_target['doa_label'][idx_ov1] = torch.stack((
            #     lams_y_doa * batch_target['doa_label'][idx_ov1][:, :, 0],
            #     (1. - lams_y_doa) * batch_target['doa_label'][new_idx_ov1][:, :, 0], 
            #     torch.zeros_like(batch_target['doa_label'][idx_ov1][:, :, 0])
            #     ), dim=2)
            batch_target['doa_label'][idx_ov1] = torch.stack((
                batch_target['doa_label'][idx_ov1][:, :, 0],
                batch_target['doa_label'][new_idx_ov1][:, :, 0], 
                torch.zeros_like(batch_target['doa_label'][idx_ov1][:, :, 0])
                ), dim=2)
        elif label_key == 'accdoa_label':
            batch_target[label_key][idx_ov1] = lams_y * batch_target[label_key][idx_ov1] + \
                (1 - lams_y) * batch_target[label_key][new_idx_ov1]
        elif label_key == 'adpit_label':
            label_idx_ov1 = batch_target[label_key][idx_ov1]
            label_idx_ov1_new = batch_target[label_key][new_idx_ov1]
            assert label_idx_ov1[:, :, 1:].sum() == 0, 'label_idx_ov1 has more than 1 source'
            new_label = torch.zeros_like(label_idx_ov1)
            new_label[:, :, :, 0] = lams_y[:, 0] * label_idx_ov1[:, :, :, 0] + (1 - lams_y[:, 0]) * label_idx_ov1_new[:, :, :, 0]
            new_label[:, :, :, 1:] = label_idx_ov1[:, :, :, 1:] + label_idx_ov1_new[:, :, :, 1:]
            B_idx, T_idx, C_idx = torch.nonzero(label_idx_ov1[:, :, 0, 0] * label_idx_ov1_new[:, :, 0, 0], as_tuple=True)
            new_label[B_idx, T_idx] = 0.
            new_label[B_idx, T_idx, 1, 0, C_idx] = lams_y[B_idx].squeeze() * label_idx_ov1[B_idx, T_idx, 0, 0, C_idx]
            new_label[B_idx, T_idx, 1, 1:, C_idx] = label_idx_ov1[B_idx, T_idx, 0, 1:, C_idx]
            new_label[B_idx, T_idx, 2, 0, C_idx] = (1 - lams_y[B_idx].squeeze()) * label_idx_ov1_new[B_idx, T_idx, 0, 0, C_idx]
            new_label[B_idx, T_idx, 2, 1:, C_idx] = label_idx_ov1_new[B_idx, T_idx, 0, 1:, C_idx]
            batch_target[label_key][idx_ov1] = new_label
        batch_target['ov'] = np.array(batch_target['ov'])
        batch_target['ov'][idx_ov1] = ['2'] * N
        batch_target['ov'] = list(batch_target['ov'])

        return batch_x, batch_target