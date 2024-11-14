import torch 
import torch.nn.functional as F
import random 
import numpy as np


class FreqShift:
    """
    This data augmentation random shift the spectrogram up or down.
    """
    def __init__(self, p=0.5, shift_range=None, direction=None, mode='reflect'):
        self.shift_range = shift_range
        self.direction = direction
        self.mode = mode
        self.p = p

    def __call__(self, batch_x, batch_target):
        N, _, _, F_dim = batch_x.shape
        for n in range(N):
            if self.p > np.random.uniform():
                batch_x[n] = self.transform(batch_x[n], F_dim)
        return batch_x, batch_target

    def transform(self, x, F_dim):
        if self.shift_range is None:
            self.shift_range = int(F_dim * 0.08)
        shift_len = torch.randint(self.shift_range, ())
        if self.direction is None:
            direction = random.choice(['up','down'])
        else:
            direction = self.direction
        new_spec = x
        if direction == 'up':
            new_spec = F.pad(new_spec, (shift_len, 0), mode=self.mode)[:, :, :F_dim]
        else:
            new_spec = F.pad(new_spec, (0, shift_len), mode=self.mode)[:, :, shift_len:]
        
        return new_spec
