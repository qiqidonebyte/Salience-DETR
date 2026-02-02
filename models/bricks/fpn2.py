# Filename: fpn_marine_enhanced.py

import torch
from torch import nn
import torch.nn.functional as F
import math


class MarineEnhancedFPN(nn.Module):
    """
    海洋增强FPN模块，专门为海洋目标检测设计
    可以直接替换原始FPN，无需修改后续neck代码
    """

    def __init__(self, features_channels, out_channels, use_marine_enhance=True):
        super(MarineEnhancedFPN, self).__init__()

        self.use_marine_enhance = use_marine_enhance
        self.out_channels = out_channels

        # 原始的上采样层
        self.up_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_channels in features_channels
        ])

        # 海洋特征增强模块
        if use_marine_enhance:
            # 海水特征提取器
            self.marine_feat_extractors = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=8),
                    nn.Conv2d(out_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                for _ in range(len(features_channels))
            ])

            # 海洋上下文注意力
            self.marine_context_attention = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(out_channels, out_channels // 8, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels // 8, out_channels, 1),
                    nn.Sigmoid()
                )
                for _ in range(len(features_channels))
            ])

            # 海洋特征融合门控
            self.marine_fusion_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels, 1),
                    nn.Sigmoid()
                )
                for _ in range(len(features_channels) - 1)  # 除了第一层
            ])

        # 横向连接层
        self.lateral = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        print(f"MarineEnhancedFPN initialized:")
        print(f"  Input channels: {features_channels}")
        print(f"  Output channels: {out_channels}")
        print(f"  Marine enhancement: {use_marine_enhance}")

    def extract_marine_context(self, x, level):
        """提取海洋上下文特征"""
        if not self.use_marine_enhance:
            return x

        # 提取海洋特定特征
        marine_feat = self.marine_feat_extractors[level](x)

        # 计算海洋上下文注意力
        context_att = self.marine_context_attention[level](marine_feat)

        # 应用上下文注意力
        enhanced_feat = x + marine_feat * context_att

        return enhanced_feat

    def fuse_marine_features(self, current_feat, prev_feat, level):
        """融合海洋特征"""
        if not self.use_marine_enhance or level == 0:
            return prev_feat

        # 获取融合门控
        gate_idx = level - 1
        fusion_gate = self.marine_fusion_gates[gate_idx]

        # 计算融合权重
        gate_input = torch.cat([current_feat, prev_feat], dim=1)
        fusion_weight = fusion_gate(gate_input)

        # 门控融合
        fused_feat = fusion_weight * current_feat + (1 - fusion_weight) * prev_feat

        return fused_feat

    def forward(self, features):
        """
        前向传播
        输入: features (dict) - backbone输出的特征字典
        输出: fpn_features (dict) - 与输入格式相同的FPN特征字典
        """
        # 兼容不同的backbone特征键
        if isinstance(features, dict):
            # 检测常见的backbone输出格式
            if 'layer2' in features and 'layer3' in features and 'layer4' in features:
                # ResNet格式
                feature_list = [features['layer2'], features['layer3'], features['layer4']]
                feature_keys = ['layer2', 'layer3', 'layer4']
            elif 'features.3' in features and 'features.5' in features and 'features.7' in features:
                # Swin Transformer格式
                feature_list = [features['features.3'], features['features.5'], features['features.7']]
                feature_keys = ['features.3', 'features.5', 'features.7']
            else:
                # 通用处理：取最后三个特征
                all_keys = list(features.keys())
                feature_keys = all_keys[-3:]  # 取最后三层
                feature_list = [features[k] for k in feature_keys]
        else:
            # 如果输入是列表，转换为字典格式处理
            feature_list = features
            feature_keys = [f'layer{i + 2}' for i in range(len(feature_list))]

        fpn_features = []

        # 自底向上处理特征
        for i, (feature, key) in enumerate(zip(feature_list, feature_keys)):
            # 1. 通道调整
            fpn_feat = self.up_layers[i](feature)

            # 2. 海洋特征增强
            if self.use_marine_enhance:
                fpn_feat = self.extract_marine_context(fpn_feat, i)

            # 3. 特征融合（从高层到低层）
            if i > 0:
                # 获取上一层特征
                prev_fpn_feat = fpn_features[i - 1]

                # 上采样对齐尺寸
                prev_fpn_feat_resized = F.interpolate(
                    prev_fpn_feat,
                    size=fpn_feat.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

                # 海洋特征融合
                if self.use_marine_enhance:
                    prev_fpn_feat_resized = self.fuse_marine_features(
                        fpn_feat, prev_fpn_feat_resized, i
                    )

                # 特征相加
                fpn_feat = fpn_feat + prev_fpn_feat_resized

            fpn_features.append(fpn_feat)

        # 4. 应用横向连接
        fpn_features = [self.lateral(f) for f in fpn_features]

        # 5. 构建输出字典（保持与输入相同的键名）
        fpn_output = {f'fpn_{key}': feat for key, feat in zip(feature_keys, fpn_features)}

        return fpn_output


class MarineAwareNeck(nn.Module):
    """
    海洋感知的neck模块
    可以增强海水背景的抑制和小目标的增强
    """

    def __init__(self, in_channels, num_levels=3):
        super(MarineAwareNeck, self).__init__()

        self.num_levels = num_levels

        # 海洋感知模块
        self.marine_aware_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=4),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels)
            )
            for _ in range(num_levels)
        ])

        # 自适应海洋特征权重
        self.marine_weight_predictors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // 4, 1),
                nn.Sigmoid()
            )
            for _ in range(num_levels)
        ])

        # 特征融合
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 1)
            for _ in range(num_levels)
        ])

    def forward(self, features):
        """
        前向传播
        输入: features (dict) - FPN输出的特征字典
        输出: enhanced_features (dict) - 增强后的特征
        """
        enhanced_features = {}

        for i, (key, feat) in enumerate(features.items()):
            if i >= self.num_levels:
                break

            # 提取海洋感知特征
            marine_feat = self.marine_aware_modules[i](feat)

            # 计算海洋权重
            marine_weight = self.marine_weight_predictors[i](feat).view(-1, 1, 1, 1)

            # 特征融合
            fused_feat = feat + marine_feat * marine_weight

            # 最终卷积
            enhanced_feat = self.fusion_convs[i](fused_feat)

            enhanced_features[f'neck_{key}'] = enhanced_feat

        return enhanced_features


# 测试兼容性
if __name__ == "__main__":
    print("=== 测试与原始代码的兼容性 ===")

    # 模拟原始backbone输出
    dummy_backbone_features = {
        'layer2': torch.rand(2, 256, 64, 64),  # batch_size=2
        'layer3': torch.rand(2, 512, 32, 32),
        'layer4': torch.rand(2, 1024, 16, 16)
    }

    print(f"Backbone输出特征形状:")
    for k, v in dummy_backbone_features.items():
        print(f"  {k}: {v.shape}")

    # 创建海洋增强FPN
    fpn = MarineEnhancedFPN(
        features_channels=[256, 512, 1024],
        out_channels=256,
        use_marine_enhance=True
    )

    # 模拟原始调用流程
    print("\n=== 模拟原始调用流程 ===")

    # 1. backbone提取特征
    multi_level_feats = dummy_backbone_features
    print(f"Backbone输出: {list(multi_level_feats.keys())}")

    # 2. FPN处理
    multi_level_feats = fpn(multi_level_feats)
    print(f"FPN输出: {list(multi_level_feats.keys())}")
    for k, v in multi_level_feats.items():
        print(f"  {k}: {v.shape}")

    # 3. Neck处理（可选）
    neck = MarineAwareNeck(in_channels=256, num_levels=3)
    multi_level_feats = neck(multi_level_feats)
    print(f"\nNeck输出: {list(multi_level_feats.keys())}")
    for k, v in multi_level_feats.items():
        print(f"  {k}: {v.shape}")

    # 测试Swin backbone格式
    print("\n=== 测试Swin Backbone格式 ===")
    swin_features = {
        'features.3': torch.rand(2, 256, 64, 64),
        'features.5': torch.rand(2, 512, 32, 32),
        'features.7': torch.rand(2, 1024, 16, 16)
    }

    fpn_output = fpn(swin_features)
    print(f"Swin FPN输出: {list(fpn_output.keys())}")

    # 测试直接集成到salience-detr
    print("\n=== 集成到salience-detr的示例 ===")


    class DemoSalienceDetr(nn.Module):
        """演示如何集成到salience-detr"""

        def __init__(self):
            super(DemoSalienceDetr, self).__init__()

            # 模拟backbone
            self.backbone = nn.Identity()  # 用恒等映射代替真实backbone

            # 海洋增强FPN
            self.fpn = MarineEnhancedFPN(
                features_channels=[256, 512, 1024],
                out_channels=256,
                use_marine_enhance=True
            )

            # 海洋感知neck
            self.neck = MarineAwareNeck(
                in_channels=256,
                num_levels=3
            )

            # 后续的detr头
            self.detr_head = nn.Identity()

        def forward(self, images):
            # 1. Backbone特征提取
            multi_level_feats = self.backbone(images)

            # 2. FPN特征金字塔
            multi_level_feats = self.fpn(multi_level_feats)

            # 3. Neck进一步处理
            multi_level_feats = self.neck(multi_level_feats)

            # 4. 输入到detr头
            output = self.detr_head(multi_level_feats)

            return output


    # 创建模型
    model = DemoSalienceDetr()

    # 模拟输入
    dummy_images = {
        'tensors': torch.rand(2, 3, 224, 224)
    }

    # 前向传播
    output = model(dummy_images)
    print("模型前向传播成功！")
    print(f"最终输出类型: {type(output)}")
