import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 物理光学增强模块 (MSRCR简化版) - 性能优化版本（精度无损）
class MSRCREnhanced(nn.Module):
    def __init__(self, scales=[15, 80, 250], weights=[1.0, 1.0, 1.0], use_separable=True, downsample_threshold=float('inf')):
        """
        Args:
            scales: 多尺度高斯核的尺度参数
            weights: 各尺度的权重
            use_separable: 是否使用可分离卷积（默认True，数学上等价于2D卷积，无精度损失，但速度快很多）
            downsample_threshold: 当sigma大于此值时，使用下采样加速（默认inf，即禁用下采样以保证精度）
                                 如果追求速度可以设置为100，但会损失一些精度
        """
        super().__init__()
        self.scales = scales
        self.weights = nn.Parameter(torch.tensor(weights).view(1, -1, 1, 1), requires_grad=False)
        self.use_separable = use_separable
        self.downsample_threshold = downsample_threshold
        
        # 创建1D高斯核（用于可分离卷积）
        # 注意：可分离卷积在数学上完全等价于2D卷积，不会损失精度
        # 因为高斯核可以分解为 G(x,y) = G(x) * G(y)
        self.gaussian_kernels_1d = nn.ParameterList([
            self._create_gaussian_kernel_1d(s) for s in scales
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
        # 形状: [3, 1, 1, kernel_size] 用于水平卷积
        # 形状: [3, 1, kernel_size, 1] 用于垂直卷积
        return kernel_1d
    
    def _create_gaussian_kernel_2d(self, sigma):
        """创建2D高斯核（原始方法，较慢）"""
        kernel_1d = self._create_gaussian_kernel_1d(sigma)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d[None, None, ...].repeat(3, 1, 1, 1)
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
            kernel_1d_down = kernel_1d_down.view(1, 1, 1, -1)
            
            # 水平卷积
            padding_h = kernel_size_down // 2
            blurred_h = F.conv2d(x_down, kernel_1d_down, padding=(0, padding_h), groups=C)
            # 垂直卷积
            blurred_h = blurred_h.transpose(2, 3)
            blurred = F.conv2d(blurred_h, kernel_1d_down, padding=(0, padding_h), groups=C)
            blurred = blurred.transpose(2, 3)
            
            # 上采样回原始尺寸
            blurred = F.interpolate(blurred, size=(H, W), mode='bilinear', align_corners=False)
        else:
            # 标准可分离卷积
            kernel_1d = kernel_1d.view(1, 1, 1, -1).to(x.device)
            padding = kernel_1d.shape[-1] // 2
            
            # 水平卷积: [B, C, H, W] -> [B, C, H, W]
            blurred = F.conv2d(x, kernel_1d, padding=(0, padding), groups=C)
            # 垂直卷积: 需要转置
            blurred = blurred.transpose(2, 3)  # [B, C, W, H]
            blurred = F.conv2d(blurred, kernel_1d, padding=(0, padding), groups=C)
            blurred = blurred.transpose(2, 3)  # [B, C, H, W]
        
        return blurred

    def forward(self, x):
        # x: input image [B, C, H, W]
        retinex_outputs = []
        
        for i, sigma in enumerate(self.scales):
            if self.use_separable:
                # 使用可分离卷积（快速）
                blurred = self._gaussian_blur_separable(x, self.gaussian_kernels_1d[i], sigma)
            else:
                # 使用原始2D卷积（慢）
                kernel = self.gaussian_kernels_2d[i].to(x.device)
                blurred = F.conv2d(x, kernel, padding=kernel.size(-1)//2, groups=3)
            
            retinex = torch.log(x + 1e-6) - torch.log(blurred + 1e-6)
            retinex_outputs.append(retinex)
        
        msr = torch.stack(retinex_outputs, dim=1)
        msr = (msr * self.weights.to(x.device)).sum(dim=1)
        # 简易色彩恢复
        mean_per_channel = x.mean(dim=(2,3), keepdim=True)
        color_restore = (x / (mean_per_channel + 1e-6)).clamp_max(10.0)
        enhanced = msr * color_restore
        return enhanced