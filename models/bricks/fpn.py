# Filename: fpn.py

import torch
from torch import nn
import torch.nn.functional as F

class FPN(nn.Module):
    """
    A simple Feature Pyramid Network module for use in object detection models.
    """

    def __init__(self, features_channels, out_channels):
        """
        Initializes the FPN module.

        :param features_channels: A list of integers representing the number of channels
                                  for each feature map from the backbone.
        :param out_channels: An integer representing the number of output channels
                             for all feature maps.
        """
        super(FPN, self).__init__()
        # Define the upsampling layers for each feature map
        self.up_layers = nn.ModuleList([
            nn.Conv2d(features_channels[i], out_channels, kernel_size=1)
            for i in range(len(features_channels))
        ])
        
        # Define the lateral connection layers
        self.lateral_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        # Define the downsampling layer
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        """
        Computes the FPN output.

        :param features: A list of feature maps from the backbone in order of
                         C2, C3, C4, C5 (from high to low resolution).
        :return: A list of FPN feature maps.
        """
        # Apply upsampling layers
        print(f"Length of layers: {len(self.up_layers)}")
        p5 = self.up_layers[2](features[2])
        p4 = self.up_layers[1](features[1])
        p3 = self.up_layers[0](features[0])

        # 调整空间尺寸以匹配特定的尺寸
        target_size = features[0].shape[-2:]  # 假设我们希望尺寸与C3特征图相同
        p4 = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)

        # Apply lateral connections
        p4 += F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 += F.interpolate(p4, scale_factor=2, mode='nearest')

        # Apply downsampling
        p4 = self.downsample(p4)

        # Return the FPN feature maps
        return [p3, p4, p5]

# Example usage:
if __name__ == "__main__":
    # Dummy input features (C2, C3, C4, C5) with random channels
    dummy_input = [torch.rand(1, 256, 64, 64), 
                   torch.rand(1, 512, 32, 32), 
                   torch.rand(1, 1024, 16, 16), 
                   torch.rand(1, 2048, 8, 8)]
    
    # Initialize FPN with the channels of the input features and desired output channels
    fpn = FPN(features_channels=[256, 512, 1024, 2048], out_channels=256)
    
    # Get FPN output
    fpn_output = fpn(dummy_input)
    print("FPN Output shape:", [o.shape for o in fpn_output])