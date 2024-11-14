import torch

class Crop:
    def __init__(self, T=8, F=8, mC=2, mask_value=0.):
        self.T = T
        self.F = F
        self.mC = mC
        self.mask_value = mask_value
    
    def __call__(self, batch_x, batch_target):
        N, C, T_dim, F_dim = batch_x.shape

        device = batch_x.device
        dtype = batch_x.dtype

        value_t = torch.rand((self.mC, N, C), device=device, dtype=dtype) * self.T
        min_value_t = torch.rand((self.mC, N, C), device=device, dtype=dtype) * (T_dim - value_t)
        value_f = torch.rand((self.mC, N, C), device=device, dtype=dtype) * self.F
        min_value_f = torch.rand((self.mC, N, C), device=device, dtype=dtype) * (F_dim - value_f)
        
        # Create broadcastable mask
        mask_start_t = min_value_t.long()[..., None, None]
        mask_end_t = (min_value_t.long() + value_t.long())[..., None, None]
        mask_t = torch.arange(T_dim, device=device, dtype=dtype)[None, None, None, :, None]
        mask_start_f = min_value_f.long()[..., None, None]
        mask_end_f = (min_value_f.long() + value_f.long())[..., None, None]
        mask_f = torch.arange(F_dim, device=device, dtype=dtype)[None, None, None, None, :]
        mask = (((mask_t >= mask_start_t) & (mask_t < mask_end_t)) & ((mask_f >= mask_start_f) & (mask_f < mask_end_f)))
        mask = torch.any(mask, dim=0)
        batch_x = batch_x.masked_fill(mask, self.mask_value)
        
        return batch_x, batch_target
