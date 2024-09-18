# Filename: fpn.py

import torch
from torch import nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, features_channels, out_channels):
        super(FPN, self).__init__()
        self.up_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in features_channels
        ])
        print(f"features_channels:{features_channels}")
        # 假设所有层都使用相同的out_channels进行横向连接
        self.lateral = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, features):
        # print(f"features: {features.keys()}")
        feature_list = [features['layer2'], features['layer3'], features['layer4']]
        # swin-l backbone的feature
        # feature_list = [features['features.3'], features['features.5'], features['features.7']]
        # 应用上采样层
        fpn_features = []
        for i, feature in enumerate(feature_list):
            # 应用卷积层调整通道数
            # print(f"feature shape:{feature.shape}")
            fpn_feature = self.up_layers[i](feature)
            if i > 0:
                # 确保上采样的特征图与 fpn_feature 的尺寸相同
                prev_fpn_feature = fpn_features[i-1]
                scaled_prev_fpn_feature = F.interpolate(
                    prev_fpn_feature, size=fpn_feature.shape[2:], mode='bilinear', align_corners=True
                )
                fpn_feature += scaled_prev_fpn_feature
            fpn_features.append(fpn_feature)
            

        # 应用横向连接层
        fpn_features = [self.lateral(f) for f in fpn_features]

        # return fpn_features[::-1]  # 反转列表以保持原始顺序
        fpn_output = {key: fpn_features[i] for i, key in enumerate(sorted(features))}
        return fpn_output

# Example usage:
if __name__ == "__main__":
    dummy_input = [torch.rand(1, 256, 64, 64), 
                   torch.rand(1, 512, 32, 32), 
                   torch.rand(1, 1024, 16, 16)]  # 输入特征图列表，长度为3
    fpn = FPN(features_channels=[256, 512, 1024], out_channels=256)  # FPN初始化
    fpn_output = fpn(dummy_input)  # 获取FPN输出
    print("FPN Output shape:", [o.shape for o in fpn_output])  # 打印FPN输出形状
