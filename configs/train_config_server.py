import os
from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# ========== 服务器训练配置（800x1333，48 epoch，ResNet50）==========
num_epochs = 48
batch_size = 1  # total_batch_size = #GPU x batch_size
num_workers = 4  # 按服务器资源调整，若不足可降到 2
pin_memory = True
print_freq = 50
starting_epoch = 0
max_norm = 0.1

output_dir = None  # None 时默认 checkpoints/{model_name}
find_unused_parameters = False

# 数据集路径（服务器）
coco_path = '/data/seadronessee/Downloads/Uncompressed Version'
train_transform = presets.detr
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/images/train",
    ann_file=f"{coco_path}/annotations/instances_train.json",
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/images/val",
    ann_file=f"{coco_path}/annotations/instances_val.json",
    transforms=None,  # eval_transform 在模型内
)

# 模型配置
model_path = "configs/salience_detr/salience_detr_resnet50_800_1333.py"
# 如需切换 Swin 版本，改为：
# model_path = "configs/salience_detr/salience_detr_swin_l_800_1333.py"

# checkpoint 恢复（默认从头），也可用环境变量 RESUME_FROM_CHECKPOINT 指定
resume_from_checkpoint = None
if "RESUME_FROM_CHECKPOINT" in os.environ:
    env_value = os.environ["RESUME_FROM_CHECKPOINT"].strip()
    if env_value:
        resume_from_checkpoint = env_value

# 优化器与学习率策略
learning_rate = 1e-4
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
# 调整 milestone 到后半程（24, 36），保持 48 epoch 训练
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[24, 36], gamma=0.1)

# 参数分组
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)

