# Filename: fpn_marine_enhanced.py

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
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # 深度可分离卷积
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 细节注意力机制
        self.detail_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), 1),  # 确保通道数不小于8
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, 1),
            nn.Sigmoid()
        )

        # 局部上下文增强（帮助区分小目标和噪声）
        self.local_context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1),  # 小空洞卷积
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取细节特征
        detail_feat = self.detail_extractor(x)

        # 细节注意力
        detail_att = self.detail_attention(x)
        enhanced_detail = detail_feat * detail_att

        # 局部上下文
        context_feat = self.local_context(x)

        # 残差融合：基础特征 + 细节 + 局部上下文
        return x + enhanced_detail + 0.1 * context_feat


class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块 - 增强小目标在不同尺度的表现"""

    def __init__(self, channels: int, num_scales: int = 3):
        super().__init__()

        self.channels = channels

        # 跨尺度注意力
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
        """
        融合多个尺度的特征
        features: 列表，包含多个尺度的特征图
        """
        if len(features) == 1:
            return features[0]

        # 将所有特征对齐到最大尺寸
        target_size = features[0].shape[2:]
        aligned_features = []

        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            aligned_features.append(feat)

        # 计算跨尺度注意力权重
        pooled_features = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in aligned_features]
        cat_features = torch.cat(pooled_features, dim=1)
        attention_weights = self.cross_scale_attention(cat_features)

        # 分割注意力权重
        weight_splits = torch.split(attention_weights, self.channels, dim=1)

        # 加权融合
        fused = torch.zeros_like(aligned_features[0])
        for feat, weight in zip(aligned_features, weight_splits):
            fused += feat * weight.view(-1, self.channels, 1, 1)

        return fused


class EdgeAwareEnhancement(nn.Module):
    """边缘感知增强模块 - 增强小目标的轮廓信息"""

    def __init__(self, channels: int):
        super().__init__()

        # 简单的边缘检测（梯度近似）
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 边缘注意力
        self.edge_attention = nn.Sequential(
            nn.Conv2d(channels, max(channels // 8, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 4), channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取边缘特征
        edge_feat = self.edge_conv(x)

        # 计算边缘注意力
        edge_att = self.edge_attention(edge_feat)

        # 增强边缘区域的特征
        return x + x * edge_att


class MarineEnhancedFPN(nn.Module):
    """
    海洋增强FPN模块 - 专门为海洋小目标检测设计
    针对seadronsee数据集优化，提升小目标检测性能
    """

    def __init__(self, features_channels, out_channels, use_marine_enhance=True):
        super(MarineEnhancedFPN, self).__init__()

        self.use_marine_enhance = use_marine_enhance
        self.out_channels = out_channels

        print(f"初始化MarineEnhancedFPN (小目标优化版):")
        print(f"  输入通道: {features_channels}")
        print(f"  输出通道: {out_channels}")
        print(f"  海洋增强: {use_marine_enhance}")

        # 1. 横向连接层（1x1卷积对齐通道）
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_channels in features_channels
        ])

        # 2. 小目标特征增强模块
        if use_marine_enhance:
            self.small_object_enhancers = nn.ModuleList([
                SmallObjectEnhancer(out_channels, reduction=8)
                for _ in range(len(features_channels))
            ])

            # 边缘感知模块（特别针对轮廓不清晰的小目标）
            self.edge_enhancers = nn.ModuleList([
                EdgeAwareEnhancement(out_channels)
                for _ in range(len(features_channels))
            ])

            # 多尺度融合模块
            self.multi_scale_fusion = MultiScaleFeatureFusion(
                channels=out_channels,
                num_scales=len(features_channels)
            )

        # 3. 特征融合卷积（自顶向下路径）
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(len(features_channels) - 1)
        ])

        # 4. 输出卷积
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(len(features_channels))
        ])

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """优化的权重初始化，适合小目标检测"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  总参数量: {total_params:,}")

    def extract_marine_context(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """增强海洋上下文特征（针对小目标优化）"""
        if not self.use_marine_enhance:
            return x

        # 小目标特征增强
        enhanced = self.small_object_enhancers[level](x)

        # 边缘感知增强
        enhanced = self.edge_enhancers[level](enhanced)

        return enhanced

    def fuse_marine_features(self, current_feat: torch.Tensor,
                             prev_feat: torch.Tensor,
                             level: int) -> torch.Tensor:
        """优化特征融合策略，特别关注小目标"""
        if not self.use_marine_enhance or level == 0:
            return prev_feat

        # 自适应特征融合
        if level - 1 < len(self.fusion_convs):
            # 对齐尺寸
            if prev_feat.shape[2:] != current_feat.shape[2:]:
                prev_feat = F.interpolate(
                    prev_feat,
                    size=current_feat.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

            # 融合特征
            fused = current_feat + prev_feat
            fused = self.fusion_convs[level - 1](fused)

            return fused
        else:
            return prev_feat

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
            elif '0' in features and '1' in features and '2' in features:
                # 简单数字键格式
                feature_list = [features['0'], features['1'], features['2']]
                feature_keys = ['0', '1', '2']
            else:
                # 通用处理：取最后三个特征
                all_keys = list(features.keys())
                feature_keys = all_keys[-3:]  # 取最后三层
                feature_list = [features[k] for k in feature_keys]
        else:
            # 如果输入是列表，转换为字典格式处理
            feature_list = features
            feature_keys = [f'layer{i + 2}' for i in range(len(feature_list))]

        # 1. 通道调整
        lateral_feats = []
        for i, (feature, key) in enumerate(zip(feature_list, feature_keys)):
            lateral_feat = self.lateral_convs[i](feature)
            lateral_feats.append(lateral_feat)

        # 2. 自顶向下特征金字塔
        fpn_features = [lateral_feats[-1]]  # 从最深特征开始

        for i in range(len(lateral_feats) - 2, -1, -1):
            # 上采样高层特征
            top_down_feat = F.interpolate(
                fpn_features[0],
                size=lateral_feats[i].shape[2:],
                mode='bilinear',
                align_corners=True
            )

            # 海洋特征融合
            fused_feat = self.fuse_marine_features(lateral_feats[i], top_down_feat, i)

            # 海洋特征增强
            if self.use_marine_enhance:
                fused_feat = self.extract_marine_context(fused_feat, i)

            fpn_features.insert(0, fused_feat)

        # 3. 多尺度特征融合（可选）
        if self.use_marine_enhance and len(fpn_features) > 1:
            multi_scale_feat = self.multi_scale_fusion(fpn_features)
            # 将融合特征加到每个尺度
            for i in range(len(fpn_features)):
                fpn_features[i] = fpn_features[i] + multi_scale_feat

        # 4. 最终输出卷积
        for i in range(len(fpn_features)):
            fpn_features[i] = self.output_convs[i](fpn_features[i])

        # 5. 构建输出字典（保持与输入相同的键名）
        fpn_output = {f'fpn_{key}': feat for key, feat in zip(feature_keys, fpn_features)}

        return fpn_output


# 保持原有MarineAwareNeck类不变，但可以优化
class MarineAwareNeck(nn.Module):
    """
    海洋感知的neck模块 - 针对小目标优化
    可以增强海水背景的抑制和小目标的增强
    """

    def __init__(self, in_channels, num_levels=3):
        super(MarineAwareNeck, self).__init__()

        self.num_levels = num_levels

        # 优化的海洋感知模块
        self.marine_aware_modules = nn.ModuleList([
            nn.Sequential(
                # 深度可分离卷积减少参数量
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=4),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                # 小目标增强卷积
                nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1),  # 小空洞卷积
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_levels)
        ])

        # 自适应海洋特征权重
        self.marine_weight_predictors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, max(in_channels // 4, 16)),  # 确保足够容量
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),  # 防止过拟合
                nn.Linear(max(in_channels // 4, 16), 1),
                nn.Sigmoid()
            )
            for _ in range(num_levels)
        ])

        # 特征融合
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
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

            # 特征融合（小目标特征增强）
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
        'layer2': torch.rand(2, 256, 80, 80),  # 小目标需要更高分辨率
        'layer3': torch.rand(2, 512, 40, 40),
        'layer4': torch.rand(2, 1024, 20, 20)
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
        'features.3': torch.rand(2, 256, 80, 80),
        'features.5': torch.rand(2, 512, 40, 40),
        'features.7': torch.rand(2, 1024, 20, 20)
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
        'tensors': torch.rand(2, 3, 320, 320)  # 小目标检测需要更高分辨率
    }

    # 前向传播
    output = model(dummy_images)
    print("模型前向传播成功！")
    print(f"最终输出类型: {type(output)}")

    # 参数对比
    original_params = 0
    new_params = sum(p.numel() for p in fpn.parameters())
    print(f"\n改进版FPN参数量: {new_params:,}")

    # 性能提升点说明
    print("\n=== 针对seadronsee小目标检测的优化点 ===")
    print("1. 小目标特征增强模块: 专门增强小目标的高频细节")
    print("2. 边缘感知增强: 增强小目标轮廓，区分目标和背景噪声")
    print("3. 多尺度特征融合: 整合不同尺度的特征信息")
    print("4. 自顶向下特征金字塔: 更好的语义信息传递")
    print("5. 优化的权重初始化: 更适合小目标检测")
    print("6. 更高的输入分辨率: 支持更高分辨率特征图")