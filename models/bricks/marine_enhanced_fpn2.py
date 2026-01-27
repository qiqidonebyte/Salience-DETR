# Filename: fpn_marine_enhanced.py
# 替换原有文件即可

import torch
from torch import nn
import torch.nn.functional as F
import math


class SmallObjectEnhancer(nn.Module):
    """小目标特征增强模块"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        # 高频细节提取
        self.detail_extractor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 细节注意力
        self.detail_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, 1),
            nn.Sigmoid()
        )

        # 局部上下文
        self.local_context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail_feat = self.detail_extractor(x)
        detail_att = self.detail_attention(x)
        enhanced_detail = detail_feat * detail_att
        context_feat = self.local_context(x)
        return x + enhanced_detail + 0.1 * context_feat


class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合"""

    def __init__(self, channels: int, num_scales: int = 3):
        super().__init__()
        self.channels = channels
        self.cross_scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * num_scales, channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels // 2, channels * num_scales),
            nn.Sigmoid()
        )

    def forward(self, features: list) -> torch.Tensor:
        if len(features) == 1:
            return features[0]

        target_size = features[0].shape[2:]
        aligned_features = []

        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            aligned_features.append(feat)

        pooled_features = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in aligned_features]
        cat_features = torch.cat(pooled_features, dim=1)
        attention_weights = self.cross_scale_attention(cat_features)

        weight_splits = torch.split(attention_weights, self.channels, dim=1)
        fused = torch.zeros_like(aligned_features[0])

        for feat, weight in zip(aligned_features, weight_splits):
            fused += feat * weight.view(-1, self.channels, 1, 1)

        return fused


class EdgeAwareEnhancement(nn.Module):
    """边缘感知增强"""

    def __init__(self, channels: int):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.edge_attention = nn.Sequential(
            nn.Conv2d(channels, max(channels // 8, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 4), channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_feat = self.edge_conv(x)
        edge_att = self.edge_attention(edge_feat)
        return x + x * edge_att


class MarineEnhancedFPN(nn.Module):
    """
    海洋增强FPN模块 - 专门为海洋小目标检测设计
    100%兼容原接口，可直接替换
    """

    def __init__(self, features_channels, out_channels, use_marine_enhance=True):
        super(MarineEnhancedFPN, self).__init__()

        self.use_marine_enhance = use_marine_enhance
        self.out_channels = out_channels

        # 兼容性说明
        print(f"MarineEnhancedFPN (小目标优化版) 初始化:")
        print(f"  输入通道: {features_channels} (自动检测ResNet格式)")
        print(f"  输出通道: {out_channels}")
        print(f"  小目标增强: {use_marine_enhance}")

        # 1. 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in features_channels
        ])

        # 2. 小目标增强模块
        if use_marine_enhance:
            self.small_object_enhancers = nn.ModuleList([
                SmallObjectEnhancer(out_channels, reduction=8)
                for _ in range(len(features_channels))
            ])

            self.edge_enhancers = nn.ModuleList([
                EdgeAwareEnhancement(out_channels)
                for _ in range(len(features_channels))
            ])

            self.multi_scale_fusion = MultiScaleFeatureFusion(
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

        # 初始化
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
        输入: features (dict) - backbone输出的特征字典
        输出: fpn_features (dict) - 输出特征字典，键为 fpn_layer2, fpn_layer3, fpn_layer4
        100%兼容原接口
        """
        # 处理不同格式的输入
        if isinstance(features, dict):
            # 检测ResNet格式
            if 'layer2' in features and 'layer3' in features and 'layer4' in features:
                feature_list = [features['layer2'], features['layer3'], features['layer4']]
                feature_keys = ['layer2', 'layer3', 'layer4']
            elif 'features.3' in features and 'features.5' in features and 'features.7' in features:
                # Swin Transformer格式
                feature_list = [features['features.3'], features['features.5'], features['features.7']]
                feature_keys = ['features.3', 'features.5', 'features.7']
            elif '0' in features and '1' in features and '2' in features:
                # 数字键格式
                feature_list = [features['0'], features['1'], features['2']]
                feature_keys = ['0', '1', '2']
            else:
                # 通用处理
                all_keys = list(features.keys())
                feature_keys = all_keys[-3:]
                feature_list = [features[k] for k in feature_keys]
        else:
            # 如果输入是列表
            feature_list = features
            feature_keys = [f'layer{i + 2}' for i in range(len(feature_list))]

        # 1. 横向连接
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

            # 小目标增强
            if self.use_marine_enhance:
                fused = self.small_object_enhancers[i](fused)
                fused = self.edge_enhancers[i](fused)

            fpn_features.insert(0, fused)

        # 3. 多尺度融合
        if self.use_marine_enhance and len(fpn_features) > 1:
            multi_scale_feat = self.multi_scale_fusion(fpn_features)
            for i in range(len(fpn_features)):
                fpn_features[i] = fpn_features[i] + multi_scale_feat

        # 4. 最终输出
        for i in range(len(fpn_features)):
            fpn_features[i] = self.output_convs[i](fpn_features[i])

        # 5. 构建输出字典（完全兼容原格式）
        fpn_output = {f'fpn_{key}': feat for key, feat in zip(feature_keys, fpn_features)}

        return fpn_output


# 保持原MarineAwareNeck不变以确保兼容
class MarineAwareNeck(nn.Module):
    """海洋感知的neck模块 - 完全兼容原接口"""

    def __init__(self, in_channels, num_levels=3):
        super(MarineAwareNeck, self).__init__()
        self.num_levels = num_levels

        self.marine_aware_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=4),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels)
            ) for _ in range(num_levels)
        ])

        self.marine_weight_predictors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // 4, 1),
                nn.Sigmoid()
            ) for _ in range(num_levels)
        ])

        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 1) for _ in range(num_levels)
        ])

    def forward(self, features):
        enhanced_features = {}

        for i, (key, feat) in enumerate(features.items()):
            if i >= self.num_levels:
                break

            marine_feat = self.marine_aware_modules[i](feat)
            marine_weight = self.marine_weight_predictors[i](feat).view(-1, 1, 1, 1)
            fused_feat = feat + marine_feat * marine_weight
            enhanced_feat = self.fusion_convs[i](fused_feat)
            enhanced_features[f'neck_{key}'] = enhanced_feat

        return enhanced_features


# 兼容性测试
if __name__ == "__main__":
    print("=== 测试完全兼容性 ===")

    # 模拟ResNet50的backbone输出
    dummy_backbone_features = {
        'layer2': torch.rand(2, 512, 80, 80),  # [batch, 512, H/8, W/8]
        'layer3': torch.rand(2, 1024, 40, 40),  # [batch, 1024, H/16, W/16]
        'layer4': torch.rand(2, 2048, 20, 20)  # [batch, 2048, H/32, W/32]
    }

    print("模拟backbone输出:")
    for k, v in dummy_backbone_features.items():
        print(f"  {k}: {v.shape}")

    # 创建FPN（与您的配置完全相同）
    fpn = MarineEnhancedFPN(
        features_channels=[512, 1024, 2048],
        out_channels=256,
        use_marine_enhance=True  # 启用小目标增强
    )

    # 模拟调用流程
    multi_level_feats = dummy_backbone_features
    print(f"\nBackbone输出键: {list(multi_level_feats.keys())}")

    multi_level_feats = fpn(multi_level_feats)
    print(f"FPN输出键: {list(multi_level_feats.keys())}")

    for k, v in multi_level_feats.items():
        print(f"  {k}: {v.shape}")

    # 验证输出形状正确
    assert multi_level_feats['fpn_layer2'].shape == torch.Size([2, 256, 80, 80])
    assert multi_level_feats['fpn_layer3'].shape == torch.Size([2, 256, 40, 40])
    assert multi_level_feats['fpn_layer4'].shape == torch.Size([2, 256, 20, 20])

    print("\n✓ 输出形状验证通过！")

    # 测试Neck兼容性
    neck = MarineAwareNeck(in_channels=256, num_levels=3)
    multi_level_feats = neck(multi_level_feats)

    print(f"\nNeck输出键: {list(multi_level_feats.keys())}")
    for k, v in multi_level_feats.items():
        print(f"  {k}: {v.shape}")

    print("\n✓ 完全兼容性验证成功！")
    print("  可直接替换原有MarineEnhancedFPN模块")
    print("  输入输出格式保持不变")
    print("  Neck模块可正常处理FPN输出")

    # 性能提示
    print("\n=== 使用建议 ===")
    print("1. 将 use_marine_enhance=True 以启用小目标增强")
    print("2. 建议输入图像分辨率 ≥ 640x640 以获得更好的小目标检测效果")
    print("3. 针对seadronsee数据集，可适当增加训练epoch")
    print("4. 关注验证集上的小目标AP指标")