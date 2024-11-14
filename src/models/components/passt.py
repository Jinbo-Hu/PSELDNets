import math
import torch
import torch.nn as nn
from models.components.utils import trunc_normal_, lecun_normal_
from models.components.model_utilities import PatchEmbed, Mlp, DropPath
from functools import partial
from collections import OrderedDict
import warnings

first_RUN = True

PLUS1_TRICK = False

def _init_vit_weights(module: nn.Module, name: str = '', 
                      head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4) # (qkv, B, num_heads, N, C//num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if PLUS1_TRICK:
            # +1 trick
            attn = torch.cat([attn, torch.zeros(attn.shape[:-1]+(1,), dtype=attn.dtype, device=attn.device)], dim=-1)
        attn = attn.softmax(dim=-1)
        if PLUS1_TRICK:
            # +1 trick
            attn = attn[...,:-1]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PaSST(nn.Module):
    """

    Based on the implementation of Vision Transformer in timm library.
     Take a look at the get_model function, adapting the weights of pretrained imagenet models.

    """

    def __init__(self, in_chans=7, u_patchout=0, s_patchout_t=0, s_patchout_f=0, 
                 img_size=(128, 998), patch_size=16, stride=10, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                 representation_size=None, distilled=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 qkv_bias=True, embed_layer=PatchEmbed, norm_layer=None, act_layer=None, weight_init='',):
        """
        Args:
            u_patchout: Unstructured Patchout integer, number of items to be removed from the final sequence
            s_patchout_t: structured Patchout time integer, number of columns to be removed from the patches grid
            s_patchout_f: structured Patchout Frequency integer, number of rows to be removed from the patches grid
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.in_chans = in_chans
        self.u_patchout = u_patchout
        self.s_patchout_t = s_patchout_t
        self.s_patchout_f = s_patchout_f
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.shape = None
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, patch_stride=stride, 
            in_chans=in_chans, embed_dim=embed_dim, flatten=False, padding=True)
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # PaSST
        # refer to https://arxiv.org/abs/2110.05069 Section 2
        self.new_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))  # for C and D tokens
        self.freq_new_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, grid_size[0], 1))  # | f
        self.time_new_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, grid_size[1]))  # __ t
        ####
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Sequential(nn.LayerNorm(self.num_features),
                                  nn.Identity())
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        trunc_normal_(self.new_pos_embed, std=.02)
        trunc_normal_(self.freq_new_pos_embed, std=.02)
        trunc_normal_(self.time_new_pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            raise RuntimeError("Not supported yet")
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'new_pos_embed', 'freq_new_pos_embed', 'time_new_pos_embed', 'cls_token', 'dist_token'}

    def forward_features(self, x):
        global first_RUN  # not jit friendly? use trace instead
        x = self.patch_embed(x.transpose(-1, -2))  # -> [B, C, F, T] 
        B_dim, E_dim, F_dim, T_dim = x.shape  # slow
        if first_RUN: print(" patch_embed : ", x.shape) # [32, 768, 6, 100]
        # Adding Time/Freq information
        if first_RUN: print(" self.time_new_pos_embed.shape", self.time_new_pos_embed.shape) # [1, 768, 1, 100]
        time_new_pos_embed = self.time_new_pos_embed
        if x.shape[-1] <= time_new_pos_embed.shape[-1]:
            if self.training:
                toffset = torch.randint(1 + time_new_pos_embed.shape[-1] - x.shape[-1], (1,)).item()
                if first_RUN: print(f" CUT with randomoffset={toffset} time_new_pos_embed.shape",
                                    time_new_pos_embed.shape) # [1, 768, 1, 100]
                time_new_pos_embed = time_new_pos_embed[:, :, :, toffset:toffset + x.shape[-1]]
            else:
                time_new_pos_embed = time_new_pos_embed[:, :, :, :x.shape[-1]]
            if first_RUN: print(" CUT time_new_pos_embed.shape", time_new_pos_embed.shape) # [1, 768, 1, 100]
        else:
            warnings.warn(
                f"the patches shape:{x.shape} are larger than the expected time encodings {time_new_pos_embed.shape}, x will be cut")
            x = x[:, :, :, :time_new_pos_embed.shape[-1]]
        x = x + time_new_pos_embed
        if first_RUN: print(" self.freq_new_pos_embed.shape", self.freq_new_pos_embed.shape) # [1, 768, 6, 1]
        x = x + self.freq_new_pos_embed

        # Structured Patchout https://arxiv.org/abs/2110.05069 Section 2.2
        if self.training and self.s_patchout_t:
            if first_RUN: print(f"X Before time Patchout of {self.s_patchout_t} ", x.size())
            # ([1, 768, 1, 82])
            random_indices = torch.randperm(T_dim)[:T_dim - self.s_patchout_t].sort().values
            x = x[:, :, :, random_indices]
            if first_RUN: print("X after time Patchout", x.size())
        if self.training and self.s_patchout_f:
            if first_RUN: print(f"X Before Freq Patchout of {self.s_patchout_f} ", x.size())
            # [1, 768, 6, 1]
            random_indices = torch.randperm(F_dim)[:F_dim - self.s_patchout_f].sort().values
            x = x[:, :, random_indices, :]
            if first_RUN: print(" \n X after freq Patchout: ", x.size())
        ###
        # Flatten the sequence
        x = x.flatten(2).transpose(1, 2) # -> [B, F*T, C]
        # Unstructured Patchout
        if first_RUN: print("X flattened", x.size()) # [32, 600, 768]
        if self.training and self.u_patchout:
            seq_len = x.shape[1]
            random_indices = torch.randperm(seq_len)[:seq_len - self.u_patchout].sort().values
            x = x[:, random_indices, :]
            if first_RUN: print("X After Unstructured Patchout", x.size())
        ####
        # Add the C/D tokens
        if first_RUN: print(" self.new_pos_embed.shape", self.new_pos_embed.shape) # [1, 2, 768]
        cls_tokens = self.cls_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, :1, :]
        if first_RUN: print(" self.cls_tokens.shape", cls_tokens.shape) # [32, 1, 768]
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            dist_token = self.dist_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, 1:, :]
            if first_RUN: print(" self.dist_token.shape", dist_token.shape) # [32, 1, 768]
            x = torch.cat((cls_tokens, dist_token, x), dim=1)

        if first_RUN: print(" final sequence x", x.shape) # [32, 602, 768]
        x = self.pos_drop(x)
        x = self.blocks(x)
        if first_RUN: print(f" after {len(self.blocks)} atten blocks x", x.shape) # [32, 602, 768]
        x = self.norm(x)
        if self.dist_token is None:
            feature = self.pre_logits(x[:, 0])
            feature_map = x[:, 1:]
        else:
            feature = x[:, :2]
            feature_map = x[:, 2:]
        _F_dim = F_dim-self.s_patchout_f if self.training else F_dim
        feature_map = feature_map.transpose(-1, -2).contiguous()
        feature_map = feature_map.reshape(B_dim, E_dim, _F_dim, T_dim).mean(2).permute(0, 2, 1)
        if first_RUN: print(" feature", feature.shape) # [32, 2, 768]
        if first_RUN: print(" feature_map", feature_map.shape) # [32, 100, 768]

        return feature, feature_map

    def forward(self, x):
        global first_RUN
        if first_RUN: print("x", x.size()) # [32, 7, 1001, 64]

        x = self.forward_features(x)
        features, feature_map = x

        if self.head_dist is not None:
            features = torch.mean(features, dim=1)
            x = self.head(feature_map)
            if first_RUN: print("head", x.size()) # [32, 100, 768]
            first_RUN = False
            return x, features
        else:
            x = self.head(feature_map)
        if first_RUN: print("head", x.size())

        first_RUN = False
        
        return x, features
    
    def forward_before(self, x):
        x = self.patch_embed(x.transpose(-1, -2))
        self.shape = B_dim, E_dim, F_dim, T_dim = x.shape  # slow
        # Adding Time/Freq information
        time_new_pos_embed = self.time_new_pos_embed
        if x.shape[-1] <= time_new_pos_embed.shape[-1]:
            if self.training:
                toffset = torch.randint(1 + time_new_pos_embed.shape[-1] - x.shape[-1], (1,)).item()
                time_new_pos_embed = time_new_pos_embed[:, :, :, toffset:toffset + x.shape[-1]]
            else:
                time_new_pos_embed = time_new_pos_embed[:, :, :, :x.shape[-1]]
        else:
            warnings.warn(
                f"the patches shape:{x.shape} are larger than the expected time encodings {time_new_pos_embed.shape}, x will be cut")
            x = x[:, :, :, :time_new_pos_embed.shape[-1]]
        x = x + time_new_pos_embed
        x = x + self.freq_new_pos_embed

        # Structured Patchout https://arxiv.org/abs/2110.05069 Section 2.2
        if self.training and self.s_patchout_t:
            random_indices = torch.randperm(T_dim)[:T_dim - self.s_patchout_t].sort().values
            x = x[:, :, :, random_indices]
        if self.training and self.s_patchout_f:
            random_indices = torch.randperm(F_dim)[:F_dim - self.s_patchout_f].sort().values
            x = x[:, :, random_indices, :]
        ###
        # Flatten the sequence
        x = x.flatten(2).transpose(1, 2)
        # Unstructured Patchout
        if self.training and self.u_patchout:
            seq_len = x.shape[1]
            random_indices = torch.randperm(seq_len)[:seq_len - self.u_patchout].sort().values
            x = x[:, random_indices, :]
        ####
        # Add the C/D tokens
        cls_tokens = self.cls_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, :1, :]
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            dist_token = self.dist_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, 1:, :]
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        
        x = self.pos_drop(x)
        return x
    
    def forward_after(self, x):
        B_dim, E_dim, F_dim, T_dim = self.shape

        x = self.norm(x)
        if self.dist_token is None:
            feature = self.pre_logits(x[:, 0])
            feature_map = x[:, 1:]
        else:
            feature = x[:, :2]
            feature_map = x[:, 2:]
        _F_dim = F_dim-self.s_patchout_f if self.training else F_dim
        feature_map = feature_map.transpose(-1, -2).contiguous()
        feature_map = feature_map.reshape(B_dim, E_dim, _F_dim, T_dim).mean(2).permute(0, 2, 1)

        if self.head_dist is not None:
            feature = torch.mean(feature, dim=1)
            x = self.head(feature_map)
            return x, feature
        else:
            x = self.head(feature_map)
        
        return x, feature