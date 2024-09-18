import torch
import torch.nn as nn
import torch.nn.functional as F


class BottomUpPathAugmentation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottomUpPathAugmentation, self).__init__()
        # 3x3 convolution with stride 2 for downsampling
        self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        # 3x3 convolution for feature fusion
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, top_feature):
        # Downsample the feature map
        downsampled = self.downsample_conv(x)
        # Element-wise addition of the downsampled feature and the top feature
        added = downsampled + top_feature
        # Apply convolution to the added feature maps
        fused_feature = self.fusion_conv(added)
        return fused_feature


if __name__ == '__main__':
    # Parameters for the feature map
    batch_size = 1
    in_channels = 256  # Typically the same as the number of channels in the top feature
    height = 64
    width = 64
    # Create a random feature map tensor
    feature_map = torch.rand(batch_size, in_channels, height, width)
    # Instantiate the Bottom-up Path Augmentation module with the same number of input and output channels
    bupa_module = BottomUpPathAugmentation(in_channels, in_channels)
    # Since the top feature is not provided in this example, we will create a random tensor
    # with the same shape as `feature_map` to simulate the top feature. In practice, this
    # should be the actual feature from the previous layer in the network.
    top_feature = torch.rand(batch_size, in_channels, height // 2, width // 2)  # Downsampled size
    # Apply the Bottom-up Path Augmentation module
    fused_feature_map = bupa_module(feature_map, top_feature)
    # Print the shape of the fused feature map to verify it's correct
    print(f"Shape of the fused feature map: {fused_feature_map.shape}")
