import torch.nn as nn
from models import accdoa
from utils.utilities import get_pylogger

log = get_pylogger(__name__)

class CRNN(accdoa.CRNN):
    def __init__(self, *args, **kwargs):
        super(CRNN, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self.num_features[-1], 
                            3 * 3 * self.num_classes, bias=True)

    def forward(self, x):
        return {
            'multi_accdoa': super().forward(x)['accdoa']
        }

class ConvConformer(accdoa.ConvConformer):
    def __init__(self, *args, **kwargs):
        super(ConvConformer, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self.num_features[-1], 
                            3 * 3 * self.num_classes, bias=True)

    def forward(self, x):
        return {
            'multi_accdoa': super().forward(x)['accdoa']
        }

class HTSAT(accdoa.HTSAT):
    def __init__(self, *args, **kwargs):
        super(HTSAT, self).__init__(*args, **kwargs)
        self.tscam_conv = nn.Conv2d(
            in_channels = self.encoder.num_features,
            out_channels = self.num_classes * 3 * 3, # NOTE: 9 for x, y and z for each class with 3 tracks
            kernel_size = (self.encoder.SF,3),
            padding = (0,1))
        self.fc = nn.Identity()
        log.info(f'Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')
        log.info(f'Non-trainable parameters: {sum(p.numel() for p in self.parameters() if not p.requires_grad)}')

    def forward(self, x):
        return {
            'multi_accdoa': super().forward(x)['accdoa']
        }

class PASST(accdoa.PASST):
    def __init__(self, *args, **kwargs):
        super(PASST, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self.encoder.num_features, 
                            3 * 3 * self.num_classes)

    def forward(self, x):
        return {
            'multi_accdoa': super().forward(x)['accdoa']
        }