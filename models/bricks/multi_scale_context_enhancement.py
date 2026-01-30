import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class ChannelAttentionEnhanced(nn.Module):
    """改进的通道注意力机制，针对海洋小目标"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        # 频率注意力分支
        self.freq_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 可学习温度参数
        self.temperature = nn.Parameter(torch.ones(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 全局平均池化
        avg_out = self.shared_mlp(self.avg_pool(x).view(B, C))
        max_out = self.shared_mlp(self.max_pool(x).view(B, C))

        # 合并
        channel_attn = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)

        # 频率注意力（对高分辨率特征）
        if H > 8 and W > 8:
            # 简化的频率特征提取
            x_pool = F.avg_pool2d(x, kernel_size=2, stride=2)
            x_dct = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=False)
            freq_attn = self.freq_attention(x_dct)

            # 自适应融合
            alpha = torch.sigmoid(self.temperature)
            channel_attn = channel_attn * (1 - alpha) + freq_attn * alpha

        return channel_attn


class SpatialAttentionEnhanced(nn.Module):
    """改进的空间注意力机制，针对海洋小目标"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # 坐标注意力
        self.x_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.y_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)

        # 局部特征提取
        self.local_conv = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, groups=channels // 4)

        # 全局上下文
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 特征融合
        self.combine_conv = nn.Sequential(
            nn.Conv2d(channels // 4 * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 坐标注意力
        x_avg = F.adaptive_avg_pool2d(x, 1)
        x_attn = self.x_conv(x_avg)
        y_attn = self.y_conv(x_avg)

        # 局部特征
        x_local = self.local_conv(x)

        # 全局特征
        x_global = self.global_conv(x)
        x_global = F.interpolate(x_global, size=(H, W), mode='bilinear', align_corners=False)

        # 合并
        x_combined = torch.cat([x_local, x_global], dim=1)
        spatial_attn = self.combine_conv(x_combined)

        return spatial_attn


class SmallTargetEnhancement(nn.Module):
    """专门的小目标增强模块"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # 高分辨率特征保持
        self.hr_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1,
                      groups=channels // 2, bias=False),
            nn.GroupNorm(8, channels // 2),
            nn.ReLU(inplace=True)
        )

        # 细节增强
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 2, kernel_size=3, padding=1, bias=False)
        )

        # 边缘增强
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # 输出卷积
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(16, channels),
            nn.ReLU(inplace=True)
        )

        # 残差连接
        self.shortcut = nn.Conv2d(channels, channels, kernel_size=1, bias=False) \
            if channels != channels else nn.Identity()

    def forward(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """scale_idx: 0表示最高分辨率（P2/P3），用于小目标"""
        identity = x

        # 高分辨率特征提取
        x_hr = self.hr_branch(x)

        # 细节增强（对高分辨率特征更有效）
        if scale_idx <= 1:  # P2/P3 层
            x_detail = self.detail_enhance(x_hr)
            x_hr = x_hr + 0.5 * x_detail

        # 边缘增强
        if scale_idx <= 1:
            # Sobel-like 边缘检测
            edge_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                         device=x.device).view(1, 1, 3, 3).float()
            edge_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                         device=x.device).view(1, 1, 3, 3).float()

            edge_x = F.conv2d(x_hr, edge_kernel_x.repeat(x_hr.shape[1], 1, 1, 1),
                              padding=1, groups=x_hr.shape[1])
            edge_y = F.conv2d(x_hr, edge_kernel_y.repeat(x_hr.shape[1], 1, 1, 1),
                              padding=1, groups=x_hr.shape[1])
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

            x_edge = self.edge_enhance(edge)
            x_hr = x_hr + 0.3 * x_edge

        # 输出
        x_out = self.output_conv(x_hr)
        x_out = x_out + self.shortcut(identity)

        return F.relu(x_out)


class MultiScaleContextEnhancement(nn.Module):
    """多尺度上下文增强模块，专门针对海洋小目标设计"""

    def __init__(
            self,
            in_channels: int = 256,
            out_channels: int = 256,
            num_scales: int = 3,
            use_dcn: bool = True,
            use_channel_attn: bool = True,
            use_spatial_attn: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales

        # 1. 可变形卷积（如果可用）
        if use_dcn:
            try:
                from mmcv.ops import DeformConv2d
                self.deform_conv = DeformConv2d(
                    in_channels, out_channels,
                    kernel_size=3, padding=1, stride=1
                )
                print("使用可变形卷积 (Deformable Conv)")
            except ImportError:
                self.deform_conv = nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, padding=1, stride=1, bias=False
                )
                print("使用标准卷积 (Deformable Conv 不可用)")
        else:
            self.deform_conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, stride=1, bias=False
            )

        # 2. 多尺度空洞卷积
        self.multi_scale_convs = nn.ModuleList()
        dilation_rates = [1, 3, 5]  # 不同尺度的空洞卷积

        for dilation in dilation_rates[:num_scales]:
            conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=3,
                          padding=dilation, dilation=dilation,
                          groups=in_channels // 4, bias=False),
                nn.GroupNorm(max(1, in_channels // 16), in_channels // 4),
                nn.ReLU(inplace=True)
            )
            self.multi_scale_convs.append(conv)

        # 3. 注意力机制
        if use_channel_attn:
            self.channel_attn = ChannelAttentionEnhanced(in_channels)
        else:
            self.channel_attn = None

        if use_spatial_attn:
            self.spatial_attn = SpatialAttentionEnhanced(in_channels)
        else:
            self.spatial_attn = None

        # 4. 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 4 * len(self.multi_scale_convs),
                      out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, out_channels // 16), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, out_channels // 16), out_channels),
            nn.ReLU(inplace=True)
        )

        # 5. 小目标增强分支
        self.small_target_branch = SmallTargetEnhancement(in_channels)

        # 6. 残差连接
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()

        # 7. 高分辨率增强
        self.hr_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, out_channels // 16), out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """处理多尺度特征图"""
        enhanced_features = []

        for i, feat in enumerate(features):
            B, C, H, W = feat.shape

            # 原始特征
            x = feat

            # 分支1: 可变形卷积
            x_deform = self.deform_conv(x)

            # 分支2: 多尺度上下文提取
            multi_scale_features = []
            for conv in self.multi_scale_convs:
                conv_feat = conv(x)
                # 自适应调整大小
                if conv_feat.shape[-2:] != (H, W):
                    conv_feat = F.interpolate(conv_feat, size=(H, W), mode='bilinear', align_corners=False)
                multi_scale_features.append(conv_feat)

            # 分支3: 小目标增强
            x_small = self.small_target_branch(x, scale_idx=i)

            # 合并多尺度特征
            if multi_scale_features:
                x_multi_scale = torch.cat(multi_scale_features, dim=1)
            else:
                x_multi_scale = x

            # 合并所有特征
            x_combined = torch.cat([x_deform, x_multi_scale, x_small], dim=1)

            # 特征融合
            x_fused = self.fusion_conv(x_combined)

            # 注意力机制
            if self.channel_attn is not None:
                channel_attn = self.channel_attn(x_fused)
                x_fused = x_fused * channel_attn

            if self.spatial_attn is not None:
                spatial_attn = self.spatial_attn(x_fused)
                x_fused = x_fused * spatial_attn

            # 残差连接
            x_final = F.relu(x_fused + self.residual(x))

            # 对高分辨率特征图进行额外增强
            if H * W > 32 * 32:  # 高分辨率特征图
                x_final = self.enhance_high_resolution(x_final)

            enhanced_features.append(x_final)

        return enhanced_features

    def enhance_high_resolution(self, x: torch.Tensor) -> torch.Tensor:
        """对高分辨率特征图进行额外增强"""
        B, C, H, W = x.shape

        # 局部上下文增强
        if H * W > 64 * 64:  # 非常高分辨率
            # 使用平均池化增强局部一致性
            x_local = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            x_enhanced = x + 0.1 * x_local

            # 细节增强
            x_detail = x_enhanced - F.avg_pool2d(x_enhanced, kernel_size=3, stride=1, padding=1)
            x_enhanced = x_enhanced + 0.05 * x_detail

            return x_enhanced
        else:
            return x


class MarineFeatureEnhancement(nn.Module):
    """海洋特征增强模块"""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels

        # 多尺度海洋特征提取
        self.marine_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False),
                nn.GroupNorm(max(1, channels // 16), channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.GroupNorm(max(1, channels // 16), channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=False),
                nn.GroupNorm(max(1, channels // 16), channels),
                nn.ReLU(inplace=True)
            )
        ])

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, channels // 16), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, channels // 16), channels),
            nn.ReLU(inplace=True)
        )

        # 海洋上下文注意力
        self.marine_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度海洋特征
        marine_features = []
        for conv in self.marine_convs:
            marine_features.append(conv(x))

        # 融合特征
        x_fused = torch.cat(marine_features, dim=1)
        x_fused = self.fusion(x_fused)

        # 海洋上下文注意力
        marine_attn = self.marine_attention(x_fused)
        x_fused = x_fused * marine_attn + x  # 残差连接

        return F.relu(x_fused)