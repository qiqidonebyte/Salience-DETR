import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 物理感知引导注意力网络 (Physics-Aware Guided Attention Network, PAGAN)
# ============================================================================
# 
# 核心思想：
#   本模块作为主干网络的轻量级并行旁路，基于Retinex物理模型生成"注意力图"，
#   告诉主网络"应该更关注原始图像的哪些区域"（如被薄雾遮挡、低可见度的区域）。
#
# 理论支撑：
#   Retinex理论假设图像 I(x,y) = R(x,y) * L(x,y)
#   - R：反射分量（物体固有属性，不受天气影响）
#   - L：光照分量（受雾、恶劣光照等天气因素影响）
#   
#   在极端天气下，L 被污染，导致目标难以识别。本模块通过多尺度Retinex分解，
#   估计哪些区域受到天气影响较大（需要更多关注），生成物理感知的注意力图。
#
# 创新点：
#   1. 物理可解释：基于Retinex理论，不是黑盒注意力
#   2. 自适应强度：根据图像质量（雾浓度等）自动调整注意力强度
#   3. 轻量高效：作为旁路分支，不增加主干网络负担
#   4. 即插即用：可与任何检测器Backbone并行使用
#
# ============================================================================

class MSRCREnhanced(nn.Module):
    """
    物理感知引导注意力模块 (Physics-Aware Guided Attention Module)
    
    输入：原始图像 [B, 3, H, W]，值域 [0, 1]
    输出：注意力图 [B, 1, H, W]，表示每个像素需要被关注的程度
    
    注意力图的含义：
    - 高值区域：受天气影响较大，需要主网络更多关注（如被雾遮挡的目标）
    - 低值区域：图像质量较好，正常处理即可
    """
    
    def __init__(self, scales=[15, 80, 250], weights=[1.0, 1.0, 1.0], use_separable=True, 
                 downsample_threshold=float('inf'), adaptive_strength=True):
        """
        Args:
            scales: 多尺度高斯核的尺度参数，用于捕获不同尺度的光照/雾霾变化
                   - 小尺度(15)：捕获局部细节的退化
                   - 中尺度(80)：捕获中等范围的雾霾
                   - 大尺度(250)：捕获全局光照变化
            weights: 各尺度的权重，控制不同尺度的贡献
            use_separable: 是否使用可分离卷积（计算效率优化，精度无损）
            downsample_threshold: 大尺度高斯模糊的加速阈值
            adaptive_strength: 是否启用自适应注意力强度（根据图像质量调整）
        """
        super().__init__()
        self.scales = scales
        self.weights = nn.Parameter(torch.tensor(weights).view(1, -1, 1, 1), requires_grad=False)
        self.use_separable = use_separable
        self.downsample_threshold = downsample_threshold
        self.adaptive_strength = adaptive_strength
        
        # ========== 自适应强度估计器 ==========
        # 功能：估计图像质量，决定注意力图的强度
        # 原理：极端天气图像通常具有低对比度、高亮度方差等特征
        # 输出：强度系数 [0, 1]，越高表示天气越恶劣，注意力越强
        if adaptive_strength:
            self.degradation_estimator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局特征
                nn.Flatten(),
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        # ========== 物理感知注意力生成器 ==========
        # 输入：Retinex特征(3) + 色彩恢复特征(3) = 6通道
        # 输出：注意力图(1通道)
        attention_in_channels = 6
        self.attention_generator = nn.Sequential(
            nn.Conv2d(attention_in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        
        # ========== 高斯核（用于Retinex分解）==========
        # 创建1D高斯核（用于可分离卷积）
        # 注意：可分离卷积在数学上完全等价于2D卷积，不会损失精度
        # 因为高斯核可以分解为 G(x,y) = G(x) * G(y)
        self.gaussian_kernels_1d = nn.ParameterList([
            nn.Parameter(self._create_gaussian_kernel_1d(s), requires_grad=False) for s in scales
        ])
        
        # 为了兼容性，也保留2D核（但只在use_separable=False时使用）
        if not use_separable:
            self.gaussian_kernels_2d = nn.ParameterList([
                self._create_gaussian_kernel_2d(s) for s in scales
            ])

    def _create_gaussian_kernel_1d(self, sigma):
        """创建1D高斯核（用于可分离卷积）"""
        kernel_size = int(2 * 3 * sigma + 1)
        # 确保是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        return kernel_1d
    
    def _create_gaussian_kernel_2d(self, sigma):
        """创建2D高斯核（原始方法，较慢）"""
        kernel_1d = self._create_gaussian_kernel_1d(sigma)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d[None, None, ...].repeat(3, 1, 1, 1)
        return kernel
    
    def _kernel_to_conv_weight(self, kernel_1d, channels, horizontal=True):
        """
        将1D核扩展成conv2d可用的权重张量，形状: [channels, 1, k, 1] 或 [channels, 1, 1, k]
        """
        if horizontal:
            kernel = kernel_1d.view(1, 1, 1, -1)
        else:
            kernel = kernel_1d.view(1, 1, -1, 1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel

    def _gaussian_blur_separable(self, x, kernel_1d, sigma):
        """
        使用可分离卷积进行高斯模糊
        将2D卷积分解为两次1D卷积，复杂度从O(k²)降到O(2k)
        
        数学原理：高斯核 G(x,y) = exp(-(x²+y²)/(2σ²)) 可以分解为
        G(x) * G(y)，其中 G(x) = exp(-x²/(2σ²))
        这在数学上完全等价，不会损失任何精度
        """
        B, C, H, W = x.shape
        
        # 对于大尺度，使用下采样加速（默认禁用以保证精度）
        if sigma > self.downsample_threshold:
            # 计算合适的下采样比例
            scale_factor = max(1.0, sigma / self.downsample_threshold)
            target_h, target_w = int(H / scale_factor), int(W / scale_factor)
            target_h = max(32, target_h)  # 确保最小尺寸
            target_w = max(32, target_w)
            
            # 下采样
            x_down = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            # 调整kernel大小以适应下采样后的图像
            kernel_size = kernel_1d.shape[0]
            kernel_size_down = max(5, int(kernel_size / scale_factor))
            if kernel_size_down % 2 == 0:
                kernel_size_down += 1
            
            # 重新创建适合下采样图像的kernel
            x_kernel = torch.arange(kernel_size_down, dtype=torch.float32, device=x.device) - kernel_size_down // 2
            kernel_1d_down = torch.exp(-x_kernel**2 / (2 * (sigma / scale_factor)**2))
            kernel_1d_down = kernel_1d_down / kernel_1d_down.sum()
            kernel_1d_down = kernel_1d_down.to(x.device)
            
            # 水平卷积
            padding_h = kernel_size_down // 2
            h_kernel = self._kernel_to_conv_weight(kernel_1d_down, C, horizontal=True)
            blurred_h = F.conv2d(x_down, h_kernel, padding=(0, padding_h), groups=C)
            v_kernel = self._kernel_to_conv_weight(kernel_1d_down, C, horizontal=False)
            blurred = F.conv2d(blurred_h, v_kernel, padding=(padding_h, 0), groups=C)
            
            # 上采样回原始尺寸
            blurred = F.interpolate(blurred, size=(H, W), mode='bilinear', align_corners=False)
        else:
            # 标准可分离卷积
            kernel_1d = kernel_1d.to(x.device)
            padding = kernel_1d.shape[0] // 2
            
            # 水平卷积: [B, C, H, W] -> [B, C, H, W]
            h_kernel = self._kernel_to_conv_weight(kernel_1d, C, horizontal=True)
            blurred = F.conv2d(x, h_kernel, padding=(0, padding), groups=C)
            v_kernel = self._kernel_to_conv_weight(kernel_1d, C, horizontal=False)
            blurred = F.conv2d(blurred, v_kernel, padding=(padding, 0), groups=C)
        
        return blurred

    def forward(self, x):
        """
        物理感知注意力生成
        
        Args:
            x: 输入图像 [B, C, H, W]，值域 [0, 1]
            
        Returns:
            attention_map: 物理感知注意力图 [B, 1, H, W]
                - 高值区域：受天气影响大，需要主网络更多关注
                - 低值区域：图像质量好，正常处理
            degradation_level: 图像退化程度 [B, 1]（可选，用于监控）
        """
        B, C, H, W = x.shape
        
        # ========== Step 1: 估计图像退化程度 ==========
        # 用于自适应调整注意力强度
        degradation_level = None
        if self.adaptive_strength:
            degradation_level = self.degradation_estimator(x)  # [B, 1]
        
        # ========== Step 2: 多尺度Retinex分解 ==========
        # 基于Retinex理论：I = R * L
        # log(I) = log(R) + log(L)
        # R = log(I) - log(L)，其中L通过高斯模糊估计
        retinex_outputs = []
        
        for i, sigma in enumerate(self.scales):
            if self.use_separable:
                blurred = self._gaussian_blur_separable(x, self.gaussian_kernels_1d[i], sigma)
            else:
                kernel = self.gaussian_kernels_2d[i].to(x.device)
                blurred = F.conv2d(x, kernel, padding=kernel.size(-1)//2, groups=3)
            
            # Retinex分解：提取反射分量
            retinex = torch.log(x + 1e-6) - torch.log(blurred + 1e-6)
            retinex_outputs.append(retinex)
        
        # 多尺度融合
        msr = torch.stack(retinex_outputs, dim=1)
        msr = (msr * self.weights.to(x.device)).sum(dim=1)  # [B, 3, H, W]
        
        # ========== Step 3: 色彩恢复特征 ==========
        # 极端天气下色彩信息也会退化，通过色彩恢复特征辅助注意力生成
        mean_per_channel = x.mean(dim=(2, 3), keepdim=True)
        color_restore = (x / (mean_per_channel + 1e-6)).clamp_max(10.0)  # [B, 3, H, W]
        
        # ========== Step 4: 生成物理感知注意力图 ==========
        # 结合Retinex特征和色彩恢复特征
        attention_features = torch.cat([msr, color_restore], dim=1)  # [B, 6, H, W]
        attention_logits = self.attention_generator(attention_features)  # [B, 1, H, W]
        
        # 基础注意力图（Sigmoid归一化到[0, 1]）
        attention_map = torch.sigmoid(attention_logits)
        
        # ========== Step 5: 自适应强度调整 ==========
        # 根据图像退化程度调整注意力强度
        # 极端天气（退化严重）-> 注意力更强
        # 正常天气（退化轻微）-> 注意力更弱，避免过度干预
        if self.adaptive_strength and degradation_level is not None:
            # degradation_level: [B, 1] -> [B, 1, 1, 1]
            strength = degradation_level.view(B, 1, 1, 1)
            # 最小强度0.3，最大强度1.0
            strength = 0.3 + 0.7 * strength
            attention_map = attention_map * strength
        
        return attention_map