from torch import optim
import torch_npu

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# 常用训练配置
num_epochs = 24  # 训练轮数
batch_size = 4  # 总批量大小 = GPU数 × batch_size
num_workers = 16  # DataLoader工作进程数
pin_memory = True  # 是否使用锁页内存
print_freq = 50  # 日志打印频率
starting_epoch = 0
max_norm = 0.1  # 梯度裁剪范数

output_dir = None  # 检查点保存路径，None时默认: checkpoints/{model_name}
find_unused_parameters = False  # 分布式训练调试有用

# 定义训练数据集
# coco_path = "/home/rjzy/PycharmProjects/data/SeaDronesSeeOD2/uncompressed"
coco_path = '/app/data/seadronessee/compressed_version'
# coco_path = '/home/rjzy/Documents/SalienceDETR/Salience-DETR/data/1000minidata'
# coco_path = '/home/rjzy/Documents/SalienceDETR/Salience-DETR/data/coco'

# 数据增强配置
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
    transforms=None,  # 评估时的变换集成在模型中
)

# 模型配置文件路径
model_path = "configs/salience_detr/salience_detr_resnet50_800_1333.py"
# model_path = "configs/salience_detr/salience_detr_swin_l_800_1333.py"

# 检查点路径（恢复训练或微调）
# resume_from_checkpoint = "checkpoints/salience_detr_resnet50_800_1333/train/2024-09-11-15_11_51/best_ap.pth"
resume_from_checkpoint = None

# 学习率和优化器配置
learning_rate = 1e-4  # 初始学习率

# 根据是否使用NPU选择优化器
use_npu = torch_npu.npu.is_available()  # 自动检测NPU

if use_npu:
    # NPU优化配置
    from apex.optimizers import FusedAdam

    optimizer = FusedAdam
    # NPU建议使用更大的权重衰减
    optimizer_kwargs = {
        'lr': learning_rate,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999)
    }
    print("检测到NPU，使用FusedAdam优化器")
else:
    # GPU/CPU优化配置
    optimizer = optim.AdamW
    optimizer_kwargs = {
        'lr': learning_rate,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999)
    }

# 学习率调度器
# 改进: 使用Cosine Annealing (参考SGDR, ICLR 2017)
# 相比MultiStepLR，Cosine Annealing通常能提升mAP 0.2-0.4%
# 对于24 epoch训练，使用Cosine Annealing
def create_lr_scheduler(optimizer):
    if num_epochs <= 24:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
        )
    else:
        # 对于更长训练，使用Cosine Annealing with Warm Restart
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
        )

lr_scheduler = create_lr_scheduler

# 定义不同参数组的学习率
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)

# NPU特定配置
if use_npu:
    # NPU训练建议配置
    num_workers = 8  # NPU建议减少工作进程数
    pin_memory = False  # NPU对锁页内存支持有限
    batch_size = 8  # NPU通常有更大内存，可以增加批量大小

    # NPU混合精度建议
    mixed_precision = 'bf16'  # NPU对bf16支持更好

    # NPU分布式训练配置
    dist_backend = 'hccl'  # NPU使用华为通信库

    print(f"NPU训练配置: workers={num_workers}, batch_size={batch_size}")