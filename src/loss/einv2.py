import torch
from .components.loss_utilities import MSELoss, BCEWithLogitsLoss, CosineLoss, L1Loss, CrossEntropyLoss
from itertools import permutations

class Losses_pit(object):
    def __init__(self, loss_fn, loss_type, method, loss_beta):
        
        if loss_fn['sed'] == 'bce':
            loss_sed_fn = BCEWithLogitsLoss(reduction='mean')
            loss_sed_fn_pit = BCEWithLogitsLoss(reduction='PIT')
        elif loss_fn['sed'] == 'ce':
            loss_sed_fn = CrossEntropyLoss(reduction='mean')
            loss_sed_fn_pit = CrossEntropyLoss(reduction='PIT')

        if loss_fn['doa'] == 'mse':
            loss_doa_fn = MSELoss(reduction='mean')
            loss_doa_fn_pit = MSELoss(reduction='PIT')
        elif loss_fn['doa'] == 'l1':
            loss_doa_fn = L1Loss(reduction='mean')
            loss_doa_fn_pit = L1Loss(reduction='PIT')
        elif loss_fn['doa'] == 'cosine':
            loss_doa_fn = CosineLoss(reduction='mean')
            loss_doa_fn_pit = CosineLoss(reduction='PIT')
        
        self.max_ov = 3
        self.beta = loss_beta
        self.loss_type = loss_type
        self.PIT_type = method
        self.losses = [loss_sed_fn, loss_doa_fn]
        self.losses_pit = [loss_sed_fn_pit, loss_doa_fn_pit]
        self.names = ['loss_all'] + [loss.name for loss in self.losses] 
        self.loss_dict_keys = ['loss_all', 'loss_sed', 'loss_doa', 'loss_other']

    def __call__(self, pred, target, epoch_it=0):
        target = {
            'sed_label': target['sed_label'][:, :, :self.max_ov, :],
            'doa_label': target['doa_label'][:, :, :self.max_ov, :],
        }
        if 'PIT' not in self.PIT_type:
            loss_sed = self.losses[0](pred['sed'], target['sed_label'])
            loss_doa = self.losses[1](pred['doa'], target['doa_label'])
        elif self.PIT_type == 'tPIT':
            loss_sed, loss_doa = self.tPIT(pred, target)
        loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa
        losses_dict = {
            'loss_all': loss_all.mean(),
            'loss_sed': loss_sed.mean(),
            'loss_doa': loss_doa.mean(),
            'loss_other': 0.
        }
        return losses_dict    
    
    def tPIT(self, pred, target):
        """Frame Permutation Invariant Training for 6 possible combinations

        Args:
            pred: {
                'sed': [batch_size, T, num_tracks=3, num_classes], 
                'doa': [batch_size, T, num_tracks=3, doas=3]
            }
            target: {
                'sed': [batch_size, T, num_tracks=3, num_classes], 
                'doa': [batch_size, T, num_tracks=3, doas=3]            
            }
        Return:
            loss_sed: Find a possible permutation to get the lowest loss of sed. 
            loss_doa: Find a possible permutation to get the lowest loss of doa. 
        """

        loss_sed_list, loss_doa_list, loss_list = [], [], []
        loss_sed, loss_doa = 0, 0
        updated_target_sed, updated_target_doa = 0, 0
        perm_list = list(permutations(range(pred['doa'].shape[2])))
        for idx, perm in enumerate(perm_list):
            loss_sed_list.append(self.losses_pit[0](pred['sed'], target['sed_label'][:, :, list(perm), :])) 
            loss_doa_list.append(self.losses_pit[1](pred['doa'], target['doa_label'][:, :, list(perm), :]))
            loss_list.append(self.beta * loss_sed_list[idx] + (1 - self.beta) * loss_doa_list[idx])
            # loss_list.append(loss_sed_list[idx]+loss_doa_list[idx])
        loss_list = torch.stack(loss_list, dim=0)
        loss_idx = torch.argmin(loss_list, dim=0)
        for idx, perm in enumerate(perm_list):
            loss_sed += loss_sed_list[idx] * (loss_idx == idx)
            loss_doa += loss_doa_list[idx] * (loss_idx == idx)
            updated_target_doa += target['doa_label'][:, :, list(perm), :] * ((loss_idx == idx)[:, :, None, None])
            updated_target_sed += target['sed_label'][:, :, list(perm), :] * ((loss_idx == idx)[:, :, None, None])
        updated_target = {
            'doa': updated_target_doa,
            'sed': updated_target_sed,
        }

        return loss_sed, loss_doa
