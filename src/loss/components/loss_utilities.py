import torch
import torch.nn as nn
import torch.nn.functional as F
eps = torch.finfo(torch.float32).eps


class MSELoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.name = 'loss_MSE'
        if self.reduction != 'PIT':
            self.loss = nn.MSELoss(reduction='mean')
        else:
            self.loss = nn.MSELoss(reduction='none')
    
    def __call__(self, pred, target):
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))

class KLDLoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.name = 'loss_MSE'
        if self.reduction != 'PIT':
            self.loss = nn.KLDivLoss(reduction='mean')
        else:
            self.loss = nn.KLDivLoss(reduction='none')
    
    def __call__(self, pred, target):
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))


class BCEWithLogitsLoss:
    def __init__(self, reduction='mean', pos_weight=None):
        self.reduction = reduction
        self.name = 'loss_BCEWithLogits'
        if self.reduction != 'PIT':
            self.loss = nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=pos_weight)
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    
    def __call__(self, pred, target):
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))


class CrossEntropyLoss:
    def __init__(self, reduction='mean', pos_weight=None):
        self.reduction = reduction
        self.name = 'loss_CrossEntropyLoss'
        if self.reduction != 'PIT':
            self.loss = nn.CrossEntropyLoss(reduction=self.reduction)
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def __call__(self, pred, target):
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))


class CosineLoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.name = 'loss_Cosine'
        self.loss = nn.CosineSimilarity(dim=-1)
    
    def __call__(self, pred, target):
        if self.reduction != 'PIT':
            return 1 - self.loss(pred, target).mean()
        else:
            return 1 - self.loss(pred, target).mean(dim=-1)


class L1Loss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.name = 'loss_L1'
        if self.reduction != 'PIT':
            self.loss = nn.L1Loss(reduction=self.reduction)
        else:
            self.loss = nn.L1Loss(reduction='none')
    
    def __call__(self, pred, target):
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))