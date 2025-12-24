# Filename: optical_fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ReflectionAttentionUnit(nn.Module):
    """
    反射注意力单元（RAU）
    基于光学物理模型：水面反射和真实物体的连线往往接近于垂直
    通过水平切割特征图并垂直比较，增强对水面反射区域的感知
    参考：Single Image Water Hazard Detection Using FCN with Reflection Attention Units [13](@ref)
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ReflectionAttentionUnit, self).__init__()
        self.in_channels = in_channels

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力（基于垂直反射特性）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 反射特征提取
        self.reflection_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca

        # 空间注意力（垂直方向特征）
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_feat)

        # 反射特征增强
        reflection_feat = self.reflection_conv(x_ca)

        # 结合注意力
        output = x_ca * sa + reflection_feat

        return output


class HighFrequencyPerception(nn.Module):
    """
    高频感知模块（HFP）
    通过DCT变换提取高频特征，增强小目标的边缘和细节
    """

    def __init__(self, in_channels):
        super(HighFrequencyPerception, self).__init__()
        self.in_channels = in_channels

        # 可学习的高通滤波器参数
        self.high_pass_weight = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)

        # 通道注意力路径
        self.channel_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力路径
        self.spatial_path = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def dct_transform(self, x):
        """修复的DCT变换实现"""
        batch, channels, height, width = x.shape
        device = x.device
        dtype = x.dtype
        
        # 使用torch.tensor的pi
        pi = torch.tensor(torch.pi, device=device, dtype=dtype)
        
        # 创建索引张量
        i_h = torch.arange(height, device=device, dtype=dtype).view(height, 1)
        j_h = torch.arange(height, device=device, dtype=dtype).view(1, height)
        
        i_w = torch.arange(width, device=device, dtype=dtype).view(width, 1)
        j_w = torch.arange(width, device=device, dtype=dtype).view(1, width)
        
        # 计算DCT基函数
        dct_basis_h = torch.cos(pi * (2 * i_h + 1) * j_h / (2 * height))
        dct_basis_w = torch.cos(pi * (2 * i_w + 1) * j_w / (2 * width))

        # 应用DCT变换
        x_reshaped = x.reshape(batch * channels, height, width)
        x_dct = torch.matmul(dct_basis_h, x_reshaped)
        x_dct = torch.matmul(x_dct, dct_basis_w.T)
        x_dct = x_dct.reshape(batch, channels, height, width)

        return x_dct

    def high_pass_filter(self, x_dct):
        """高通滤波：保留高频成分"""
        batch, channels, h, w = x_dct.shape
        device = x_dct.device
        
        # 创建高通掩码（向量化版本）
        center_h, center_w = h // 2, w // 2
        radius = min(center_h, center_w) // 4
        
        # 创建坐标网格
        y_coords = torch.arange(h, device=device).view(h, 1)
        x_coords = torch.arange(w, device=device).view(1, w)
        
        # 计算距离平方
        dist_sq = (y_coords - center_h) ** 2 + (x_coords - center_w) ** 2
        
        # 创建掩码（高频区域为1，低频区域为0）
        mask = (dist_sq > radius ** 2).float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # 扩展为 [1, 1, h, w]
        mask = mask.expand(batch, channels, h, w)  # 扩展为 [batch, channels, h, w]
        
        high_freq = x_dct * mask
        return high_freq

    def forward(self, x):
        # DCT变换
        x_dct = self.dct_transform(x)

        # 高通滤波提取高频特征
        high_freq = self.high_pass_filter(x_dct)

        # 逆DCT变换（简化实现）
        high_freq_spatial = self.dct_transform(high_freq)

        # 生成通道和空间注意力
        channel_att = self.channel_path(high_freq_spatial)
        spatial_att = self.spatial_path(high_freq_spatial)

        # 应用注意力
        x_channel = x * channel_att
        x_spatial = x_channel * spatial_att

        # 高频特征融合
        output = x_spatial + high_freq_spatial * self.high_pass_weight

        return output


class OpticalPhysicsFPN(nn.Module):
    """
    基于光学物理模型改进的FPN
    融合反射注意力单元和高频感知模块，专门针对海域小目标检测优化
    """

    def __init__(self, features_channels, out_channels, use_rau=True, use_hfp=True):
        super(OpticalPhysicsFPN, self).__init__()
        self.features_channels = features_channels
        self.out_channels = out_channels
        self.use_rau = use_rau
        self.use_hfp = use_hfp

        print(f"features_channels: {features_channels}")

        # 基础FPN上采样层
        self.up_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in features_channels
        ])

        # 横向连接层
        self.lateral = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 光学物理增强模块
        if use_rau:
            self.rau_layers = nn.ModuleList([
                ReflectionAttentionUnit(out_channels) for _ in range(len(features_channels))
            ])

        if use_hfp:
            self.hfp_layers = nn.ModuleList([
                HighFrequencyPerception(out_channels) for _ in range(len(features_channels))
            ])

        # 特征融合层（用于融合光学增强特征）
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(features_channels))
        ])

        # 海域小目标专用增强层
        self.marine_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(len(features_channels))
        ])

    def forward(self, features):
        """
        features: 字典形式，包含backbone不同层的特征
        例如: {'layer2': tensor, 'layer3': tensor, 'layer4': tensor}
        """
        # 提取特征列表（按分辨率从高到低）
        feature_list = [features['layer2'], features['layer3'], features['layer4']]

        fpn_features = []
        optical_features = []

        for i, feature in enumerate(feature_list):
            # 1. 基础FPN通道调整
            fpn_feature = self.up_layers[i](feature)

            # 2. 光学物理增强
            optical_feature = fpn_feature.clone()

            if self.use_rau:
                optical_feature = self.rau_layers[i](optical_feature)

            if self.use_hfp:
                optical_feature = self.hfp_layers[i](optical_feature)

            optical_features.append(optical_feature)

            # 3. 特征融合（原始特征 + 光学增强特征）
            if i > 0:
                # 上采样前一层的特征
                prev_fpn_feature = fpn_features[i - 1]
                scaled_prev_fpn_feature = F.interpolate(
                    prev_fpn_feature, size=fpn_feature.shape[2:],
                    mode='bilinear', align_corners=True
                )

                # 融合光学增强特征
                fused_feature = torch.cat([fpn_feature, optical_feature], dim=1)
                fused_feature = self.fusion_layers[i](fused_feature)

                # 海域小目标增强
                marine_att = self.marine_enhance[i](fused_feature)
                enhanced_feature = fused_feature * marine_att

                fpn_feature = enhanced_feature + scaled_prev_fpn_feature
            else:
                # 第一层直接融合
                fused_feature = torch.cat([fpn_feature, optical_feature], dim=1)
                fused_feature = self.fusion_layers[i](fused_feature)

                marine_att = self.marine_enhance[i](fused_feature)
                fpn_feature = fused_feature * marine_att

            fpn_features.append(fpn_feature)

        # 应用横向连接层
        fpn_features = [self.lateral(f) for f in fpn_features]

        # 返回字典格式，保持与输入一致
        fpn_output = {
            'layer2': fpn_features[0],
            'layer3': fpn_features[1],
            'layer4': fpn_features[2]
        }

        return fpn_output


class MarineOpticalLoss(nn.Module):
    """
    海域光学物理损失函数
    基于光学模型约束特征学习
    """

    def __init__(self, alpha=0.1, beta=0.05):
        super(MarineOpticalLoss, self).__init__()
        self.alpha = alpha  # 反射一致性权重
        self.beta = beta  # 高频保持权重

    def reflection_consistency_loss(self, features):
        """
        反射一致性损失：鼓励特征在垂直方向具有反射对称性
        参考水面反射的物理特性[11,13](@ref)
        """
        loss = 0
        for feat in features:
            # 垂直翻转特征
            flipped = torch.flip(feat, dims=[2])
            # 计算反射一致性损失
            loss += F.mse_loss(feat[:, :, feat.shape[2] // 2:],
                               flipped[:, :, :feat.shape[2] // 2])
        return loss / len(features)

    def high_frequency_preservation_loss(self, features, targets):
        """
        高频保持损失：确保小目标边缘特征不被平滑
        参考HS-FPN的高频感知思想[6](@ref)
        """
        loss = 0
        for feat, target in zip(features, targets):
            # 计算梯度差异（边缘保持）
            feat_grad_x = torch.abs(feat[:, :, :, 1:] - feat[:, :, :, :-1])
            feat_grad_y = torch.abs(feat[:, :, 1:, :] - feat[:, :, :-1, :])

            target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
            target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

            loss += F.l1_loss(feat_grad_x, target_grad_x) + \
                    F.l1_loss(feat_grad_y, target_grad_y)
        return loss / len(features)

    def forward(self, pred_features, target_features):
        """
        pred_features: 预测的特征金字塔输出
        target_features: 目标特征（可以是ground truth或教师网络特征）
        """
        # 基础特征重建损失
        recon_loss = sum(F.mse_loss(p, t) for p, t in zip(pred_features, target_features))

        # 光学物理约束损失
        reflection_loss = self.reflection_consistency_loss(pred_features)
        hf_loss = self.high_frequency_preservation_loss(pred_features, target_features)

        total_loss = recon_loss + self.alpha * reflection_loss + self.beta * hf_loss

        return total_loss


# 示例使用
if __name__ == "__main__":
    # 模拟输入特征
    dummy_features = {
        'layer2': torch.rand(1, 256, 64, 64),
        'layer3': torch.rand(1, 512, 32, 32),
        'layer4': torch.rand(1, 1024, 16, 16)
    }

    # 初始化光学物理FPN
    optical_fpn = OpticalPhysicsFPN(
        features_channels=[256, 512, 1024],
        out_channels=256,
        use_rau=True,
        use_hfp=True
    )

    # 前向传播
    fpn_output = optical_fpn(dummy_features)

    print("光学物理FPN输出形状:")
    for key, value in fpn_output.items():
        print(f"{key}: {value.shape}")

    # 测试损失函数
    target_features = [
        torch.rand(1, 256, 64, 64),
        torch.rand(1, 256, 32, 32),
        torch.rand(1, 256, 16, 16)
    ]

    pred_features = list(fpn_output.values())
    loss_fn = MarineOpticalLoss()
    loss = loss_fn(pred_features, target_features)
    print(f"\n海域光学物理损失: {loss.item():.4f}")
