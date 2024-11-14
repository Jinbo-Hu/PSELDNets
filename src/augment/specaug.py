import torch
from torchaudio import functional as F


class SpecAugment:
    def __init__(self, xy_ratio, T=20, F=8, mT=4, mF=2, mask_value=0.):
        self.T = T
        self.F = F
        self.mT = mT
        self.mF = mF
        self.xy_ratio = xy_ratio
        self.T_y = int(T / self.xy_ratio)
        self.mask_value = mask_value

    def __call__(self, batch_x, batch_target):
        r"""
        Args: 
            batch_x (torch.Tensor): (N, C, T, F) shape
            batch_target (dict): dictionary of target tensors (N, T, ...)
        Returns:
            batch_x (torch.Tensor): (N, C, T, F) shape
            batch_target (dict): dictionary of target tensors
        """
        
        N, C, T_dim, F_dim = batch_x.shape
        T_y_dim = int(T_dim / self.xy_ratio)

        dim = batch_x.dim()
        device = batch_x.device
        dtype = batch_x.dtype

        #### MASK in the time dimension
        # Rewrite the following code to effect on both batch_x and batch_target
        value = torch.rand((self.mT, N), device=device, dtype=dtype) * self.T_y
        min_value = torch.rand((self.mT, N), device=device, dtype=dtype) * (T_y_dim - value)
        # Create broadcastable mask for target
        mask_start = min_value.long()
        mask_end = min_value.long() + value.long()
        mask = torch.arange(0, T_y_dim, device=device, dtype=dtype)
        for key in batch_target.keys():
            if 'label' not in key:
                continue
            batch_y = batch_target[key]
            mask_start_y = mask_start.reshape((self.mT, N) + (1,) * (batch_y.dim() - 1))
            mask_end_y = mask_end.reshape((self.mT, N) + (1,) * (batch_y.dim() - 1))
            mask_y = torch.any((mask >= mask_start_y) & (mask < mask_end_y), dim=0)
            batch_y = batch_y.transpose(1, -1)
            batch_y.masked_fill_(mask_y, self.mask_value)
            batch_target[key] = batch_y.transpose(1, -1)
        # Create broadcastable mask for data
        mask_start_x = mask_start[..., None, None, None] * self.xy_ratio
        mask_end_x = mask_end[..., None, None, None] * self.xy_ratio
        mask = torch.arange(0, T_dim, device=device, dtype=dtype)
        mask_x = torch.any((mask >= mask_start_x) & (mask < mask_end_x), dim=0)
        batch_x = batch_x.transpose(2, -1)
        batch_x.masked_fill_(mask_x, self.mask_value)
        batch_x = batch_x.transpose(2, -1)    

        #### MASK in the frequency dimension
        for _ in range(self.mF):
            batch_x = F.mask_along_axis_iid(batch_x, axis=3, mask_value=self.mask_value, mask_param=self.F)

        return batch_x, batch_target
