from torch import optim
import os
import torch_npu

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# ============ NPU 检测 ============
NPU_ENABLED = torch_npu.npu.is_available()
NPU_DEVICES = os.environ.get('ASCEND_VISIBLE_DEVICES', '0').split(',')
NPU_COUNT = len(NPU_DEVICES) if NPU_DEVICES[0] else 0

print(f"NPU 可用: {NPU_ENABLED}")
if NPU_ENABLED:
    print(f"NPU 设备: {NPU_DEVICES}")
    print(f"NPU 数量: {NPU_COUNT}")

# ============ 常用训练配置 ============
num_epochs = 24
batch_size = 4   # 根据NPU数量调整batch_size
num_workers = 8 if NPU_ENABLED else 16  # NPU建议减少workers
pin_memory = False if NPU_ENABLED else True  # NPU对pin_memory支持有限
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

# ============ 训练恢复 ============
resume_from_checkpoint = None

# ============ 优化器配置 ============
learning_rate = 1e-4

# NPU使用标准AdamW，不使用Apex
# 直接在配置中使用标准AdamW
optimizer = lambda params: optim.AdamW(
    params,
    lr=learning_rate,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

print(f"使用标准AdamW优化器 (NPU兼容)")

# ============ 学习率调度器 ============
# 根据NPU数量调整milestones
if NPU_COUNT >= 4:
    milestones = [8, 16, 20]  # 多卡时更快的调度
else:
    milestones = [10, 18]  # 单卡/少卡时标准调度

lr_scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=milestones, gamma=0.1
)

# ============ 参数组 ============
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)