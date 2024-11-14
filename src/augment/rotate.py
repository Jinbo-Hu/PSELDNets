import numpy as np
import torch
import random

class Rotation:
    def __init__(self,  p, rotation_type):
        self.p = p
        self.type = rotation_type
    
    def __call__(self, batch_x, batch_target):
        N = batch_x.shape[0]
        
        for n in range(N):

            if np.random.uniform() >= self.p:
                continue

            data = batch_x[n]
            if 'accdoa_label' in batch_target.keys():
                label_key = 'accdoa_label'
                T, C = batch_target['accdoa_label'].shape[1:]
                doa = batch_target['accdoa_label'][n].reshape(T, 3, C//3).transpose(1, 2)
            elif 'doa_label' in batch_target.keys():
                label_key = 'doa_label'
                doa = batch_target['doa_label'][n]
            elif 'adpit_label' in batch_target.keys():
                label_key = 'adpit_label'
                seddoa = batch_target[label_key][n].transpose(-1, -2)
                doa = seddoa[..., 1:]
            
            if self.type == 48:
                x, y = self.transform48(data, doa)
            elif self.type == 16:
                x, y = self.transform16(data, doa)

            batch_x[n] = x
            if label_key == 'accdoa_label': 
                y = y.transpose(1, 2).reshape(T, -1)
            elif label_key == 'adpit_label':
                y = torch.cat([seddoa[..., :1], y], dim=-1)
                y = y.transpose(-1, -2)
            batch_target[label_key][n] = y
                
        return batch_x, batch_target


    def transform48(self, x, doa):
        """
        Reference: 
        'FIRST ORDER AMBISONICS DOMAIN SPATIAL AUGMENTATION FOR DNN-BASED DIRECTION OF ARRIVAL ESTIMATION', 
        Luca Mazzon, Yuma Koizumi, Masahiro Yasuda, Noboru Harada

        All coordinates should be swapped as possible as they can.

        Input:
            x: waveform (channels, time), x[1] is Y, x[2] is Z, x[3] is X
            doa: doa (T, tracks, coordinates(x, y, z))
        """
        trans_dict = {
            (0,1,2): (1,2,3), 
            (0,2,1): (2,1,3), 
            (1,0,2): (3,2,1), 
            (1,2,0): (2,3,1), 
            (2,0,1): (3,1,2), 
            (2,1,0): (1,3,2) 
            }
        perm = list(trans_dict.keys())
        xx, yy, zz = random.choice(perm)
        s_x, s_y, s_z = trans_dict[(xx, yy, zz)]
        signx, signy, signz = np.random.choice([-1, 1], size=3)
        x = torch.stack((x[0], signy*x[s_x], signz*x[s_y], signx*x[s_z]), axis=0)
        doa = torch.stack((signx*doa[..., xx], signy*doa[..., yy], signz*doa[..., zz]), axis=-1)
        
        return x, doa  
    
    def transform16(self, x, doa):
        """
        Reference: 
        'FIRST ORDER AMBISONICS DOMAIN SPATIAL AUGMENTATION FOR DNN-BASED DIRECTION OF ARRIVAL ESTIMATION', 
        Luca Mazzon, Yuma Koizumi, Masahiro Yasuda, Noboru Harada

        All coordinates should be swapped as possible as they can.

        Input:
            x: waveform (channels, time), x[1] is Y, x[2] is Z, x[3] is X
            doa: doa (T, tracks, coordinates(x, y, z))
        """
        trans_dict = {
            (0,1,2): (1,2,3), 
            (1,0,2): (3,2,1) 
            }
        perm = list(trans_dict.keys())
        xx, yy, zz = random.choice(perm)
        s_x, s_y, s_z = trans_dict[(xx, yy, zz)]
        signx, signy, signz = np.random.choice([-1, 1], size=3)
        x = torch.stack((x[0], signy*x[s_x], signz*x[s_y], signx*x[s_z]), axis=0)
        doa = torch.stack((signx*doa[..., xx], signy*doa[..., yy], signz*doa[..., zz]), axis=-1)
        
        return x, doa  
    
        
    