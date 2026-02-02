import torch
import torch_npu
from typing import Optional, Dict, Any


def setup_npu_environment(npu_id: int = 0):
    """设置NPU环境"""
    if not torch_npu.npu.is_available():
        return False

    # 设置NPU设备
    torch_npu.npu.set_device(npu_id)

    # 启用NPU性能优化
    torch_npu.npu.set_option("ACL_PRECISION_MODE", "allow_mix_precision")
    torch_npu.npu.set_option("ACL_OP_SELECT_IMPL_MODE", "high_precision")

    return True


def to_npu(tensor_or_module, npu_id: int = 0):
    """将张量或模块转移到NPU"""
    if not torch_npu.npu.is_available():
        return tensor_or_module

    if isinstance(tensor_or_module, torch.Tensor):
        return tensor_or_module.to(f'npu:{npu_id}')
    elif isinstance(tensor_or_module, torch.nn.Module):
        return tensor_or_module.to(f'npu:{npu_id}')
    else:
        return tensor_or_module


def npu_synchronize(npu_id: int = 0):
    """同步NPU设备"""
    if torch_npu.npu.is_available():
        torch_npu.npu.synchronize(npu_id)


def empty_npu_cache():
    """清空NPU缓存"""
    if torch_npu.npu.is_available():
        torch_npu.npu.empty_cache()


def get_npu_memory_info(npu_id: int = 0) -> Dict[str, Any]:
    """获取NPU内存信息"""
    if not torch_npu.npu.is_available():
        return {}

    try:
        memory_allocated = torch_npu.npu.memory_allocated(npu_id) / 1024 ** 3  # GB
        memory_cached = torch_npu.npu.memory_reserved(npu_id) / 1024 ** 3  # GB
        memory_max_allocated = torch_npu.npu.max_memory_allocated(npu_id) / 1024 ** 3  # GB

        return {
            'allocated_gb': memory_allocated,
            'cached_gb': memory_cached,
            'max_allocated_gb': memory_max_allocated
        }
    except:
        return {}


class NPUProfiler:
    """NPU性能分析器"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch_npu.npu.is_available()
        self.events = {}

    def record(self, name: str, npu_id: int = 0):
        """记录事件"""
        if self.enabled:
            self.events[name] = torch_npu.npu.Event(enable_timing=True)
            self.events[name].record()

    def elapsed_time(self, start_name: str, end_name: str) -> float:
        """计算两个事件间的时间"""
        if self.enabled and start_name in self.events and end_name in self.events:
            self.events[end_name].synchronize()
            return self.events[start_name].elapsed_time(self.events[end_name])
        return 0.0


def optimize_model_for_npu(model: torch.nn.Module, use_amp: bool = True) -> torch.nn.Module:
    """为NPU优化模型"""
    if not torch_npu.npu.is_available():
        return model

    # 转换为NPU兼容的格式
    model = model.to('npu')

    # 启用混合精度
    if use_amp:
        from torch.cuda.amp import autocast
        # 这里可以使用autocast包装模型

    return model


def save_checkpoint_for_npu(state: Dict, filename: str):
    """保存NPU检查点（处理设备映射）"""
    # 保存前将NPU张量转移到CPU
    cpu_state = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and value.is_npu:
            cpu_state[key] = value.cpu()
        else:
            cpu_state[key] = value

    torch.save(cpu_state, filename)


def load_checkpoint_for_npu(filename: str, map_to_npu: bool = True) -> Dict:
    """加载NPU检查点"""
    checkpoint = torch.load(filename, map_location='cpu')

    if map_to_npu and torch_npu.npu.is_available():
        # 将检查点映射到NPU
        npu_checkpoint = {}
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                npu_checkpoint[key] = value.to('npu')
            else:
                npu_checkpoint[key] = value
        return npu_checkpoint

    return checkpoint