# Filename: marine_enhanced_fpn3.py
# 专门针对海域小目标检测优化的FPN模块

import torch
from torch import nn
import torch.nn.functional as F
import math


class MarineAttentionModule(nn.Module):
    """海域小目标注意力模块"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        # 高频细节提取（针对小目标）
        self.high_freq_extractor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 空间注意力（定位小目标位置）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # 通道注意力（增强小目标特征）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 高频特征提取
        high_freq = self.high_freq_extractor(x)

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))

        # 通道注意力
        channel_att = self.channel_attention(x)

        # 特征增强
        enhanced = high_freq * spatial_att * channel_att
        return x + enhanced


class MultiScaleContextFusion(nn.Module):
    """多尺度上下文融合模块"""

    def __init__(self, channels: int, num_scales: int = 3):
        super().__init__()

        self.pyramid_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, dilation=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_scales)
        ])

        # 跨尺度特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: list) -> torch.Tensor:
        if len(features) == 1:
            return features[0]

        # 多尺度特征对齐和融合
        target_size = features[0].shape[2:]
        aligned_features = []

        for i, feat in enumerate(features):
            # 调整到目标尺寸
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)

            # 尺度特定处理
            feat = self.pyramid_conv[i](feat)
            aligned_features.append(feat)

        # 特征融合
        fused = torch.cat(aligned_features, dim=1)
        return self.fusion_conv(fused)


class SmallTargetEnhancer(nn.Module):
    """小目标特征增强器"""

    def __init__(self, channels: int):
        super().__init__()

        # 细节增强分支
        self.detail_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels // 4),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 上下文增强分支
        self.context_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

        # 特征选择门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail_feat = self.detail_branch(x)  # 增强细节
        context_feat = self.context_branch(x)  # 增强上下文

        # 自适应特征融合
        gate_weights = self.gate(x)
        detail_weight = gate_weights[:, 0:1]
        context_weight = gate_weights[:, 1:2]

        enhanced = detail_feat * detail_weight + context_feat * context_weight
        return x + enhanced


class MarineEnhancedFPN_v3(nn.Module):
    """
    第三代海域小目标检测FPN
    专为海洋小目标优化，保持100%兼容性
    """

    def __init__(self, features_channels, out_channels, use_marine_enhance=True):
        super().__init__()

        self.use_marine_enhance = use_marine_enhance
        self.out_channels = out_channels

        print(f"MarineEnhancedFPN_v3 (海域小目标优化版) 初始化:")
        print(f"  输入通道: {features_channels}")
        print(f"  输出通道: {out_channels}")
        print(f"  小目标增强: {use_marine_enhance}")

        # 1. 横向连接（保持原结构）
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in features_channels
        ])

        # 2. 海域小目标增强模块
        if use_marine_enhance:
            self.marine_attentions = nn.ModuleList([
                MarineAttentionModule(out_channels, reduction=8)
                for _ in range(len(features_channels))
            ])

            self.small_target_enhancers = nn.ModuleList([
                SmallTargetEnhancer(out_channels)
                for _ in range(len(features_channels))
            ])

            self.multi_scale_fusion = MultiScaleContextFusion(
                channels=out_channels,
                num_scales=len(features_channels)
            )

        # 3. 特征融合卷积
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(features_channels) - 1)
        ])

        # 4. 输出卷积
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(features_channels))
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  总参数量: {total_params:,}")

    def forward(self, features):
        """
        前向传播 - 完全兼容原接口
        """
        # 处理不同格式的输入特征
        if isinstance(features, dict):
            if 'layer2' in features and 'layer3' in features and 'layer4' in features:
                feature_list = [features['layer2'], features['layer3'], features['layer4']]
                feature_keys = ['layer2', 'layer3', 'layer4']
            else:
                all_keys = list(features.keys())
                feature_keys = all_keys[-3:]
                feature_list = [features[k] for k in feature_keys]
        else:
            feature_list = features
            feature_keys = [f'layer{i + 2}' for i in range(len(feature_list))]

        # 1. 横向连接处理
        lateral_feats = []
        for i, feature in enumerate(feature_list):
            lateral_feat = self.lateral_convs[i](feature)
            lateral_feats.append(lateral_feat)

        # 2. 自顶向下特征金字塔
        fpn_features = [lateral_feats[-1]]  # 从最深特征开始

        for i in range(len(lateral_feats) - 2, -1, -1):
            # 上采样
            top_down_feat = F.interpolate(
                fpn_features[0],
                size=lateral_feats[i].shape[2:],
                mode='bilinear',
                align_corners=True
            )

            # 特征融合
            if i < len(self.fusion_convs):
                fused = top_down_feat + lateral_feats[i]
                fused = self.fusion_convs[i](fused)
            else:
                fused = top_down_feat + lateral_feats[i]

            # 海域小目标增强
            if self.use_marine_enhance:
                fused = self.marine_attentions[i](fused)  # 注意力增强
                fused = self.small_target_enhancers[i](fused)  # 小目标增强

            fpn_features.insert(0, fused)

        # 3. 多尺度特征融合增强
        if self.use_marine_enhance and len(fpn_features) > 1:
            multi_scale_feat = self.multi_scale_fusion(fpn_features)
            # 残差连接增强
            for i in range(len(fpn_features)):
                fpn_features[i] = fpn_features[i] + 0.5 * multi_scale_feat

        # 4. 最终输出处理
        for i in range(len(fpn_features)):
            fpn_features[i] = self.output_convs[i](fpn_features[i])

        # 5. 构建输出字典（完全兼容）
        fpn_output = {f'fpn_{key}': feat for key, feat in zip(feature_keys, fpn_features)}

        return fpn_output


# 兼容性测试
if __name__ == "__main__":
    print("=== MarineEnhancedFPN_v3 兼容性测试 ===")

    # 模拟输入
    dummy_backbone_features = {
        'layer2': torch.rand(2, 512, 80, 80),
        'layer3': torch.rand(2, 1024, 40, 40),
        'layer4': torch.rand(2, 2048, 20, 20)
    }

    # 创建FPN实例
    fpn = MarineEnhancedFPN_v3(
        features_channels=[512, 1024, 2048],
        out_channels=256,
        use_marine_enhance=True
    )

    # 前向传播测试
    output = fpn(dummy_backbone_features)

    print("输出特征形状:")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")

    # 验证兼容性
    assert output['fpn_layer2'].shape == torch.Size([2, 256, 80, 80])
    assert output['fpn_layer3'].shape == torch.Size([2, 256, 40, 40])
    assert output['fpn_layer4'].shape == torch.Size([2, 256, 20, 20])

    print("✓ 完全兼容性验证通过！")
    print("✓ 可直接替换原有FPN模块")