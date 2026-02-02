"""
Exponential Moving Average (EMA) Module
参考: "Model soups: averaging weights of multiple fine-tuned models improves accuracy without 
increasing inference time" (ICML 2022) 和 "Exponential Moving Average Normalization for Self-supervised 
and Semi-supervised Learning" (CVPR 2021)

EMA可以提升模型稳定性和最终性能，通常能提升mAP 0.2-0.5%
"""
import copy
import torch
from torch import nn


class EMA(nn.Module):
    """
    Exponential Moving Average for model parameters.
    This helps stabilize training and improve final performance.
    """
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.decay = decay
        self.device = device
        
        # Create shadow copy of model parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Update shadow parameters with exponential moving average"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        """Apply shadow parameters to model"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        """Restore original parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

