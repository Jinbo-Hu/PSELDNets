from loss.components.loss_utilities import (MSELoss, L1Loss)

class Losses(object):
    def __init__(self, loss_fn, loss_type):
        super().__init__()
        
        if loss_fn == 'mse':
            self.losses =  MSELoss()
        elif loss_fn == 'l1':
            self.losses = L1Loss()
        self.loss_type = loss_type
        self.names =  [self.losses.name]
        self.loss_dict_keys = ['loss_all', 'loss_accdoa', 'loss_other']

    def __call__(self, pred, target):
        loss = self.losses(pred['accdoa'], target['accdoa_label'])
        loss_all = loss + 0.0
        losses_dict = {
            'loss_all': loss_all,
            'loss_accdoa': loss,
            'loss_other': 0.
        }
        return losses_dict