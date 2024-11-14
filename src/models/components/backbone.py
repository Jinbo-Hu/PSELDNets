import torch
import torch.nn as nn
from models.components.model_utilities import DoubleConv, ConvBlock


class CNN8(nn.Module):
    
    def __init__(self, in_channels=4, num_features=[32, 64, 128, 256], 
                 pretrained_path=None):
        super().__init__()
        
        self.conv_block1 = ConvBlock(in_channels, num_features[0],
                                    pool_size=(2, 2), pool_type='avg')
        self.conv_block2 = ConvBlock(num_features[0], num_features[1],
                                    pool_size=(2, 2), pool_type='avg')
        self.conv_block3 = ConvBlock(num_features[1], num_features[2],
                                    pool_size=(2, 2), pool_type='avg')
        self.conv_block4 = ConvBlock(num_features[2], num_features[3],
                                    pool_size=(1, 2), pool_type='avg')
        
        self.convs =[self.conv_block1, self.conv_block2, 
                     self.conv_block3, self.conv_block4]
    
    def forward(self, x):
        """
        x: spectrogram, (batch_size, num_channels, num_frames, num_freqBins)
        """
        for conv in self.convs:
            x = conv(x)
        return x


class CNN12(nn.Module):

    def __init__(self, in_channels=4, num_features=[64, 128, 256, 512, 1024, 2048]):
        super().__init__()
        
        self.conv_block1 = ConvBlock(in_channels, num_features[0],
                                     pool_size=(2, 2), pool_type='avg')
        self.conv_block2 = ConvBlock(num_features[0], num_features[1],
                                     pool_size=(2, 2), pool_type='avg')
        self.conv_block3 = ConvBlock(num_features[1], num_features[2],
                                     pool_size=(2, 2), pool_type='avg')
        self.conv_block4 = ConvBlock(num_features[2], num_features[3],
                                     pool_size=(1, 2), pool_type='avg')
        self.conv_block5 = ConvBlock(num_features[3], num_features[4],
                                     pool_size=(1, 2), pool_type='avg')
        self.conv_block6 = ConvBlock(num_features[4], num_features[5],
                                     pool_size=(1, 2), pool_type='avg')
        
        self.convs = [self.conv_block1, self.conv_block2, self.conv_block3,
                      self.conv_block4, self.conv_block5, self.conv_block6]
    
    def forward(self, x):
        """
        x: spectrogram, (batch_size, num_channels, num_frames, num_freqBins)
        """
        for conv in self.convs:
            x = conv(x)

        return x