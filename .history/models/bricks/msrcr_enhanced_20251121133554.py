import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# 1. 物理光学增强模块 (MSRCR简化版)
class MSRCREnhanced(nn.Module):
    def __init__(self, scales=[15, 80, 250], weights=[1.0, 1.0, 1.0]):
        super().__init__()
        self.scales = scales
        self.weights = torch.tensor(weights).view(1, -1, 1, 1)
        self.gaussian_kernels = [self._create_gaussian_kernel(s) for s in scales]

    def _create_gaussian_kernel(self, sigma, channels=3):
        kernel_size = int(2 * 3 * sigma + 1)
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d[None, None, ...].repeat(channels, 1, 1, 1)
        return kernel

    def forward(self, x):
        # x: input image [B, C, H, W]
        retinex_outputs = []
        for kernel in self.gaussian_kernels:
            kernel = kernel.to(x.device)
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