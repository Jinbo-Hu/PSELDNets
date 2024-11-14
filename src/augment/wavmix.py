import numpy as np
import torch
from torch.distributions.beta import Beta
import random

class WavMix:
    def __init__(self, alpha, p) -> None:
        ''' Raw wav Time-domain mixing
        Args:
            alpha: 0.0-1.0, 0.5
        '''
        self.beta = Beta(alpha, alpha)
        self.p = p

    def __call__(self, batch_x, batch_target):

        if random.random() > self.p:
            return batch_x, batch_target

        ov = np.array(batch_target['ov'])

        idx_ov1 = [n for n in range(len(ov)) if ov[n] == '1']
        idx_ov2 = [n for n in range(len(ov)) if ov[n] == '2']
        add_ov = random.choice(['1', '2'])
        if add_ov == '1':
            new_idx_ov = np.random.permutation(idx_ov1)
        elif add_ov == '2':
            new_idx_ov = np.random.permutation(idx_ov2)
        N = min(len(idx_ov1), len(new_idx_ov))

        if N == 0:
            return batch_x, batch_target
        
        label_keys = [k for k in batch_target.keys() if 'label' in k]
        label_key = label_keys[0]
        lambs = self.beta.sample((N,)).to(batch_x.device)
        # lambs = torch.ones((N,)).to(batch_x.device)
        # lambs = 0.5 * torch.ones((N,)).to(batch_x.device)
        x_shape = (N,) + (1,) * (batch_x.ndim - 1)
        lams_x = lambs.reshape(x_shape)
        if len(label_keys) == 2: # sed_label and doa_label
            y_sed_shape = (N, ) + (1,) * (batch_target['sed_label'].ndim - 2)
            y_doa_shape = (N, ) + (1,) * (batch_target['doa_label'].ndim - 2)
            lams_y_sed = lambs.reshape(y_sed_shape)
            lams_y_doa = lambs.reshape(y_doa_shape)
        else: # accdoa_label or adpit_label
            y_shape = (N, ) + (1,) * (batch_target[label_key].ndim - 1)
            lams_y = lambs.reshape(y_shape)        
        
        batch_x[idx_ov1[:N]] = lams_x * batch_x[idx_ov1[:N]] + (1. - lams_x) * batch_x[new_idx_ov[:N]]
        if len(label_keys) == 2:
            batch_target['sed_label'][idx_ov1[:N]] = torch.stack((
                lams_y_sed * batch_target['sed_label'][idx_ov1[:N]][:, :, 0],
                (1 - lams_y_sed) * batch_target['sed_label'][new_idx_ov[:N]][:, :, 0], 
                (1 - lams_y_sed) * batch_target['sed_label'][new_idx_ov[:N]][:, :, 1],
                ), dim=2)
            # batch_target['doa_label'][idx_ov1[:N]] = torch.stack((
            #     lams_y_doa * batch_target['doa_label'][idx_ov1[:N]][:, :, 0],
            #     (1 - lams_y_doa) * batch_target['doa_label'][new_idx_ov[:N]][:, :, 0], 
            #     (1 - lams_y_doa) * batch_target['doa_label'][new_idx_ov[:N]][:, :, 1]
            #     ), dim=2)
            # batch_target['sed_label'][idx_ov1[:N]] = torch.stack((
            #     batch_target['sed_label'][idx_ov1[:N]][:, :, 0], 
            #     batch_target['sed_label'][new_idx_ov[:N]][:, :, 0], 
            #     batch_target['sed_label'][new_idx_ov[:N]][:, :, 1]
            #     ), dim=2)
            batch_target['doa_label'][idx_ov1[:N]] = torch.stack((
                batch_target['doa_label'][idx_ov1[:N]][:, :, 0],
                batch_target['doa_label'][new_idx_ov[:N]][:, :, 0], 
                batch_target['doa_label'][new_idx_ov[:N]][:, :, 1]
                ), dim=2)
        elif label_key == 'accdoa_label':
            batch_target[label_key][idx_ov1[:N]] = lams_y * batch_target[label_key][idx_ov1[:N]] + \
                (1 - lams_y) * batch_target[label_key][new_idx_ov[:N]]
            # batch_target[label_key][idx_ov1[:N]] = batch_target[label_key][idx_ov1[:N]] + \
            #     batch_target[label_key][new_idx_ov[:N]]
        elif label_key == 'adpit_label':
            label_idx_ov1 = batch_target[label_key][idx_ov1[:N]]
            assert label_idx_ov1[:, :, 1:].sum() == 0, 'label_idx_ov1 has more than 1 source'
            label_new_idx_ov = batch_target[label_key][new_idx_ov[:N]]
            new_label = torch.zeros_like(label_idx_ov1)
            new_label[:, :, :, 0] = lams_y[:, 0] * label_idx_ov1[:, :, :, 0] + (1 - lams_y[:, 0]) * label_new_idx_ov[:, :, :, 0]
            new_label[:, :, :, 1:] = label_idx_ov1[:, :, :, 1:] + label_new_idx_ov[:, :, :, 1:]
            if add_ov == '1':
                assert label_new_idx_ov[:, :, 1:].sum() == 0, 'label_new_idx_ov has more than 1 source'
                B_idx, T_idx, C_idx = torch.nonzero(label_idx_ov1[:, :, 0, 0] * label_new_idx_ov[:, :, 0, 0], as_tuple=True)
                new_label[B_idx, T_idx] = 0.
                new_label[B_idx, T_idx, 1, 0, C_idx] = lams_y[B_idx].squeeze() * label_idx_ov1[B_idx, T_idx, 0, 0, C_idx]
                new_label[B_idx, T_idx, 1, 1:, C_idx] = label_idx_ov1[B_idx, T_idx, 0, 1:, C_idx]
                new_label[B_idx, T_idx, 2, 0, C_idx] = (1 - lams_y[B_idx].squeeze()) * label_new_idx_ov[B_idx, T_idx, 0, 0, C_idx]
                new_label[B_idx, T_idx, 2, 1:, C_idx] = label_new_idx_ov[B_idx, T_idx, 0, 1:, C_idx]
            elif add_ov == '2':
                assert label_new_idx_ov[:, :, 3:].sum() == 0, 'label_new_idx_ov has more than 2 source'
                # two sources from the same class
                B_idx, T_idx, C_idx = torch.nonzero(label_idx_ov1[:, :, 0, 0] * label_new_idx_ov[:, :, 0, 0], as_tuple=True)
                new_label[B_idx, T_idx, :, :, C_idx] = 0.
                new_label[B_idx, T_idx, 1, 0, C_idx] = lams_y[B_idx].squeeze() * label_idx_ov1[B_idx, T_idx, 0, 0, C_idx]
                new_label[B_idx, T_idx, 2, 0, C_idx] = (1 - lams_y[B_idx].squeeze()) * label_new_idx_ov[B_idx, T_idx, 0, 0, C_idx]
                new_label[B_idx, T_idx, 1, 1:, C_idx] = label_idx_ov1[B_idx, T_idx, 0, 1:, C_idx]
                new_label[B_idx, T_idx, 2, 1:, C_idx] = label_new_idx_ov[B_idx, T_idx, 0, 1:, C_idx]
                # three sources from the same class
                B_idx, T_idx, C_idx = torch.nonzero(label_idx_ov1[:, :, 0, 0] * label_new_idx_ov[:, :, 1, 0], as_tuple=True)
                new_label[B_idx, T_idx, :, :, C_idx] = 0.
                new_label[B_idx, T_idx, 3, 0, C_idx] = lams_y[B_idx].squeeze() * label_idx_ov1[B_idx, T_idx, 0, 0, C_idx]
                new_label[B_idx, T_idx, 3, 1:, C_idx] = label_idx_ov1[B_idx, T_idx, 0, 1:, C_idx]
                new_label[B_idx, T_idx, 4, 0, C_idx] = (1 - lams_y[B_idx].squeeze()) * label_new_idx_ov[B_idx, T_idx, 1, 0, C_idx]
                new_label[B_idx, T_idx, 4, 1:, C_idx] = label_new_idx_ov[B_idx, T_idx, 1, 1:, C_idx]
                new_label[B_idx, T_idx, 5, 0, C_idx] = (1 - lams_y[B_idx].squeeze()) * label_new_idx_ov[B_idx, T_idx, 2, 0, C_idx]
                new_label[B_idx, T_idx, 5, 1:, C_idx] = label_new_idx_ov[B_idx, T_idx, 2, 1:, C_idx]
            batch_target[label_key][idx_ov1[:N]] = new_label
        
        ov[idx_ov1[:N]] = [str(int(n)+1) for n in ov[new_idx_ov[:N]]]
            
        batch_target['ov'] = list(ov)


        return batch_x, batch_target