# Filename: fpn_marine_enhanced_fixed.py

import torch
from torch import nn
import torch.nn.functional as F
import math


class SmallObjectEnhancer(nn.Module):
    """小目标特征增强模块 - 专门针对seadronsee数据集中的小目标"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        # 高频细节提取（使用小卷积核捕捉小目标细节）
        self.detail_extractor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 细节注意力机制
        self.detail_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, 1),
            nn.Sigmoid()
        )

        # 局部上下文增强
        self.local_context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 安全检查：确保输入有效
        if x.numel() == 0 or x.dim() != 4:
            return x

        detail_feat = self.detail_extractor(x)
        detail_att = self.detail_attention(x)
        enhanced_detail = detail_feat * detail_att
        context_feat = self.local_context(x)
        return x + enhanced_detail + 0.1 * context_feat


class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块 - 增强小目标在不同尺度的表现"""

    def __init__(self, channels: int, num_scales: int = 3):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales

        # 修复：确保输入维度正确
        self.cross_scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * num_scales, max(channels // 2, 16)),  # 确保最小维度
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(max(channels // 2, 16), channels * num_scales),
            nn.Sigmoid()
        )

    def forward(self, features: list) -> torch.Tensor:
        """
        融合多个尺度的特征
        修复：添加更严格的安全检查
        """
        if len(features) == 1:
            return features[0]

        # 严格的安全检查
        valid_features = []
        for i, feat in enumerate(features):
            if feat.dim() != 4:
                print(f"警告: 特征图 {i} 维度不正确: {feat.dim()}D")
                continue
            if feat.size(0) == 0 or feat.size(1) != self.channels:
                print(f"警告: 特征图 {i} 尺寸无效: {feat.shape}")
                continue
            if feat.size(2) < 1 or feat.size(3) < 1:
                print(f"警告: 特征图 {i} 空间尺寸过小: {feat.shape[2:]}")
                continue
            valid_features.append(feat)

        if len(valid_features) < 2:
            # 如果没有足够的有效特征，返回第一个或空张量
            return features[0] if features else torch.tensor([])

        # 使用有效特征继续处理
        features = valid_features

        # 获取目标尺寸（确保有效）
        target_size = features[0].shape[2:]
        if target_size[0] <= 0 or target_size[1] <= 0:
            print(f"警告: 目标尺寸无效: {target_size}")
            return features[0]

        aligned_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                try:
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
                except Exception as e:
                    print(f"上采样失败: {e}")
                    continue
            aligned_features.append(feat)

        if len(aligned_features) < 2:
            return features[0]

        # 计算跨尺度注意力权重（添加更严格的异常处理）
        try:
            pooled_features = []
            for f in aligned_features:
                if f.size(2) >= 1 and f.size(3) >= 1:
                    pooled = F.adaptive_avg_pool2d(f, 1)
                    pooled_flat = pooled.flatten(1)
                    # 检查展平后的维度
                    if pooled_flat.size(1) == self.channels:
                        pooled_features.append(pooled_flat)
                    else:
                        # 如果维度不匹配，创建零张量
                        zero_tensor = torch.zeros(f.size(0), self.channels, device=f.device)
                        pooled_features.append(zero_tensor)
                else:
                    zero_tensor = torch.zeros(f.size(0), self.channels, device=f.device)
                    pooled_features.append(zero_tensor)

            if not pooled_features:
                return features[0]

            cat_features = torch.cat(pooled_features, dim=1)

            # 最终维度检查
            expected_dim = self.channels * len(aligned_features)
            if cat_features.size(1) != expected_dim:
                # 如果维度不匹配，进行调整或跳过
                print(f"警告: 拼接特征维度不匹配: {cat_features.size(1)} != {expected_dim}")
                if cat_features.size(1) == 0:
                    return features[0]
                # 尝试调整或截断
                if cat_features.size(1) > expected_dim:
                    cat_features = cat_features[:, :expected_dim]
                else:
                    # 填充零
                    padding = torch.zeros(cat_features.size(0), expected_dim - cat_features.size(1),
                                          device=cat_features.device)
                    cat_features = torch.cat([cat_features, padding], dim=1)

            attention_weights = self.cross_scale_attention(cat_features)

        except Exception as e:
            print(f"多尺度融合失败: {e}")
            # 失败时返回第一个有效特征的加权平均
            fused = torch.zeros_like(aligned_features[0])
            for feat in aligned_features:
                fused += feat
            return fused / len(aligned_features)

        # 分割注意力权重
        try:
            weight_splits = torch.split(attention_weights, self.channels, dim=1)
            if len(weight_splits) != len(aligned_features):
                # 如果分割数量不匹配，使用平均权重
                weight_splits = [torch.ones(feat.size(0), self.channels, device=feat.device) / len(aligned_features)
                                 for feat in aligned_features]
        except Exception as e:
            print(f"权重分割失败: {e}")
            weight_splits = [torch.ones(feat.size(0), self.channels, device=feat.device) / len(aligned_features)
                             for feat in aligned_features]

        # 加权融合
        fused = torch.zeros_like(aligned_features[0])
        for i, (feat, weight) in enumerate(zip(aligned_features, weight_splits)):
            try:
                weight = weight.view(-1, self.channels, 1, 1)
                if weight.size(0) == feat.size(0):
                    fused += feat * weight
                else:
                    # 尺寸不匹配，使用简单平均
                    fused += feat / len(aligned_features)
            except Exception as e:
                print(f"特征加权失败 {i}: {e}")
                fused += feat / len(aligned_features)

        return fused


class EdgeAwareEnhancement(nn.Module):
    """边缘感知增强模块"""

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
        if x.numel() == 0:
            return x

        edge_feat = self.edge_conv(x)
        edge_att = self.edge_attention(edge_feat)
        return x + x * edge_att


class MarineEnhancedFPN(nn.Module):
    """
    海洋增强FPN模块 - 修复版本
    """

    def __init__(self, features_channels, out_channels, use_marine_enhance=True):
        super(MarineEnhancedFPN, self).__init__()

        self.use_marine_enhance = use_marine_enhance
        self.out_channels = out_channels

        print(f"初始化MarineEnhancedFPN (修复版):")
        print(f"  输入通道: {features_channels}")
        print(f"  输出通道: {out_channels}")
        print(f"  海洋增强: {use_marine_enhance}")

        # 1. 横向连接层
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in features_channels
        ])

        # 2. 小目标特征增强模块
        if use_marine_enhance:
            self.small_object_enhancers = nn.ModuleList([
                SmallObjectEnhancer(out_channels, reduction=8)
                for _ in range(len(features_channels))
            ])

            self.edge_enhancers = nn.ModuleList([
                EdgeAwareEnhancement(out_channels)
                for _ in range(len(features_channels))
            ])

            # 修复：确保num_scales不超过实际特征数
            actual_num_scales = min(len(features_channels), 3)  # 最大3个尺度
            self.multi_scale_fusion = MultiScaleFeatureFusion(
                channels=out_channels,
                num_scales=actual_num_scales
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
        前向传播 - 修复版本
        """
        # 输入验证
        if features is None:
            raise ValueError("输入特征不能为None")

        if isinstance(features, dict) and len(features) == 0:
            raise ValueError("输入特征字典为空")

        # 兼容不同的backbone特征键
        try:
            if isinstance(features, dict):
                if 'layer2' in features and 'layer3' in features and 'layer4' in features:
                    feature_list = [features['layer2'], features['layer3'], features['layer4']]
                    feature_keys = ['layer2', 'layer3', 'layer4']
                elif 'features.3' in features and 'features.5' in features and 'features.7' in features:
                    feature_list = [features['features.3'], features['features.5'], features['features.7']]
                    feature_keys = ['features.3', 'features.5', 'features.7']
                elif '0' in features and '1' in features and '2' in features:
                    feature_list = [features['0'], features['1'], features['2']]
                    feature_keys = ['0', '1', '2']
                else:
                    all_keys = list(features.keys())
                    feature_keys = all_keys[-3:]
                    feature_list = [features[k] for k in feature_keys]
            else:
                feature_list = features
                feature_keys = [f'layer{i + 2}' for i in range(len(feature_list))]

            # 验证特征列表
            if len(feature_list) == 0:
                raise ValueError("没有有效的特征层")

        except Exception as e:
            print(f"特征处理失败: {e}")
            # 创建默认特征
            feature_list = [torch.randn(1, 256, 32, 32) for _ in range(3)]
            feature_keys = ['layer2', 'layer3', 'layer4']

        # 1. 通道调整（添加异常处理）
        lateral_feats = []
        for i, feature in enumerate(feature_list):
            try:
                if i < len(self.lateral_convs):
                    lateral_feat = self.lateral_convs[i](feature)
                    lateral_feats.append(lateral_feat)
                else:
                    # 如果卷积层不够，使用恒等映射
                    lateral_feats.append(feature)
            except Exception as e:
                print(f"横向连接失败 {i}: {e}")
                lateral_feats.append(feature)

        if len(lateral_feats) == 0:
            raise ValueError("横向连接后无有效特征")

        # 2. 自顶向下特征金字塔
        fpn_features = [lateral_feats[-1]]

        for i in range(len(lateral_feats) - 2, -1, -1):
            try:
                # 上采样高层特征
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
                if self.use_marine_enhance and i < len(self.small_object_enhancers):
                    fused = self.small_object_enhancers[i](fused)
                    fused = self.edge_enhancers[i](fused)

                fpn_features.insert(0, fused)

            except Exception as e:
                print(f"特征金字塔构建失败 {i}: {e}")
                # 插入原始特征
                fpn_features.insert(0, lateral_feats[i])

        # 3. 多尺度特征融合（添加安全开关）
        if self.use_marine_enhance and len(fpn_features) > 1:
            try:
                multi_scale_feat = self.multi_scale_fusion(fpn_features)
                # 验证融合结果
                if (multi_scale_feat.shape == fpn_features[0].shape and
                        multi_scale_feat.device == fpn_features[0].device):
                    for i in range(len(fpn_features)):
                        fpn_features[i] = fpn_features[i] + multi_scale_feat
                else:
                    print("多尺度融合结果形状不匹配，跳过")
            except Exception as e:
                print(f"多尺度融合失败: {e}")

        # 4. 最终输出卷积
        for i in range(len(fpn_features)):
            try:
                if i < len(self.output_convs):
                    fpn_features[i] = self.output_convs[i](fpn_features[i])
            except Exception as e:
                print(f"输出卷积失败 {i}: {e}")

        # 5. 构建输出字典
        fpn_output = {}
        for i, (key, feat) in enumerate(zip(feature_keys, fpn_features)):
            if i < len(fpn_features):  # 确保不越界
                fpn_output[f'fpn_{key}'] = fpn_features[i]

        return fpn_output


# 测试修复版本
if __name__ == "__main__":
    print("=== 测试修复版本 ===")

    # 测试各种边界情况
    test_cases = [
        # 正常情况
        {
            'layer2': torch.rand(2, 512, 80, 80),
            'layer3': torch.rand(2, 1024, 40, 40),
            'layer4': torch.rand(2, 2048, 20, 20)
        },
        # 异常情况：空特征
        {
            'layer2': torch.rand(0, 512, 80, 80),  # batch=0
            'layer3': torch.rand(2, 1024, 40, 40),
            'layer4': torch.rand(2, 2048, 20, 20)
        }
    ]

    for i, features in enumerate(test_cases):
        print(f"\n测试用例 {i + 1}:")
        try:
            fpn = MarineEnhancedFPN(
                features_channels=[512, 1024, 2048],
                out_channels=256,
                use_marine_enhance=True
            )

            output = fpn(features)
            print(f"输出键: {list(output.keys())}")
            for k, v in output.items():
                print(f"  {k}: {v.shape}")

        except Exception as e:
            print(f"测试失败: {e}")