from torch import optim
import os
import torch_npu

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# ============ NPU 检测 ============
NPU_ENABLED = torch_npu.npu.is_available()
if NPU_ENABLED:
    print(f"检测到NPU设备: {torch_npu.npu.device_count()} 个")
    for i in range(torch_npu.npu.device_count()):
        print(f"  NPU {i}: {torch_npu.npu.get_device_name(i)}")
else:
    print("未检测到NPU设备，将使用CPU/GPU训练")

# 获取NPU设备
NPU_DEVICES = os.environ.get('ASCEND_VISIBLE_DEVICES', '0').split(',')
NPU_COUNT = len(NPU_DEVICES) if NPU_DEVICES[0] else 0

# ============ 基础训练配置 ============
num_epochs = 24
batch_size = 4 * max(1, NPU_COUNT)  # 根据NPU数量调整
num_workers = 8
pin_memory = False  # NPU建议关闭pin_memory
print_freq = 50
starting_epoch = 0
max_norm = 0.1

output_dir = None
find_unused_parameters = False

# ============ 数据集配置 ============
coco_path = '/app/data/seadronessee/compressed_version'
train_transform = presets.detr
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/train2017",
    ann_file=f"{coco_path}/annotations/instances_train2017.json",
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/val2017",
    ann_file=f"{coco_path}/annotations/instances_val2017.json",
    transforms=None,
)

# ============ 模型配置 ============
model_path = "configs/salience_detr/salience_detr_resnet50_800_1333.py"
resume_from_checkpoint = None
learning_rate = 1e-4

# ============ 优化器配置 ============
# NPU使用标准AdamW
optimizer = lambda params: optim.AdamW(
    params,
    lr=learning_rate,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# ============ 学习率调度器 ============
lr_scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[10], gamma=0.1
)

# ============ 参数组 ============
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)