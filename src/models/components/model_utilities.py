import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import to_2tuple
from models.components.conformer import ConformerBlocks
from .model_utilities_adapt import Adapter



def get_linear_layer(method='', rir_simulate='', *args, **kwargs):
    # method = method.split('_')
    if 'lora' in method:
        from .model_utilities_adapt import Linear as LinearLoRA
        kwargs.update(kwargs.get('linear_kwargs', {}))
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}
        return LinearLoRA(*args, **kwargs)
    else:
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}
        return nn.Linear(*args, **kwargs)
    

def get_conv2d_layer(method='', rir_simulate='', **kwargs):
    # method = method.split('_')
    if 'lora' in method:
        from .model_utilities_adapt import Conv2d as Conv2dLoRA
        kwargs.update(kwargs.get('conv_kwargs', {}))
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}
        return Conv2dLoRA(**kwargs)
    else:
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}
        return nn.Conv2d(**kwargs)


class CrossStitch(nn.Module):
    def __init__(self, feat_dim):

        super().__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(feat_dim, 2, 2).uniform_(0.1, 0.9)
            )
    
    def forward(self, x, y):
        if x.dim() == 4:
            equation = 'c, nctf -> nctf'
        elif x.dim() == 3:
            equation = 'c, ntc -> ntc'
        else:
            raise ValueError('x must be 3D or 4D tensor')
        x = torch.einsum(equation, self.weight[:, 0, 0], x) + \
            torch.einsum(equation, self.weight[:, 0, 1], y)
        y = torch.einsum(equation, self.weight[:, 1, 0], x) + \
            torch.einsum(equation, self.weight[:, 1, 1], y)
        return x, y


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, bias=False,
                pool_size=(2,2), pool_type='avg'):
        super().__init__()

        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=pool_size)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        else:
            raise Exception('pool_type must be avg or max')

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self.pool,
        )
        
    def forward(self, x):
        x = self.double_conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, bias=False,
                pool_size=(2,2), pool_type='avg'):
        
        super(ConvBlock, self).__init__()

        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=pool_size)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        else:
            raise Exception('pool_type must be avg or max')
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=bias)
                            
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=bias)
                            
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.,
                 **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = get_linear_layer(in_features=in_features, 
                                    out_features=hidden_features,
                                    **kwargs)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = get_linear_layer(in_features=hidden_features, 
                                    out_features=out_features,
                                    **kwargs)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.adapter_enable = 'adapter' in kwargs.get('method', '') and \
            'MlpAdapter' in kwargs['adapt_kwargs'].get('position', '')
        if self.adapter_enable:
            adapt_kwargs = kwargs.get('adapt_kwargs')
            self.ds_adapter = adapt_kwargs.get('new_adapter', {})
            if adapt_kwargs['type'] == 'adapter':
                self.adapter = Adapter(in_features, **adapt_kwargs)
                if self.ds_adapter:
                    self.adapter_ds = Adapter(in_features, **self.ds_adapter)

    def forward(self, x):
        xs, xs_ds = 0., 0.
        if self.adapter_enable:
            xs = self.adapter(x)
            if self.ds_adapter:
                xs_ds = self.adapter_ds(x)

        x = self.fc2(self.drop(self.act(self.fc1(x))))
        
        x = x + xs + xs_ds

        x = self.drop(x)
        return x
    

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                 norm_layer=None, flatten=True, patch_stride=16, padding=True, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        if padding:
            padding = ((patch_size[0] - patch_stride[0]) // 2, 
                       (patch_size[1] - patch_stride[1]) // 2)
        else:
            padding = 0

        self.proj = get_conv2d_layer(in_channels=in_chans, out_channels=embed_dim, 
                                     kernel_size=patch_size, stride=patch_stride, 
                                     padding=padding, **kwargs)
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
        #                       stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Decoder(nn.Module):
    def __init__(self, decoder, num_feats, num_layers=2, **kwargs):
        super().__init__()
        self.num_feats = num_feats
        if decoder == 'gru':
            self.decoder = nn.GRU(input_size=num_feats, hidden_size=num_feats//2, 
                                  num_layers=num_layers, bidirectional=True, 
                                  batch_first=True, **kwargs)
        elif decoder == 'conformer':
            self.decoder = ConformerBlocks(encoder_dim=num_feats, num_layers=num_layers, **kwargs)
        elif decoder == 'transformer':
            self.decoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=num_feats, nhead=8, 
                                           batch_first=True, **kwargs),
                num_layers=num_layers)
        elif decoder is None:
            self.decoder = nn.Identity()
        else:
            raise NotImplementedError(f"{decoder} is not implemented")

    def forward(self, x):
        if isinstance(self.decoder, nn.RNNBase):
            x = self.decoder(x)[0]
        else: x = self.decoder(x)
        return x
