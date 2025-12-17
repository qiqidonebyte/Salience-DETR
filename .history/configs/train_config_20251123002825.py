from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 5  # train epochs (CPU训练建议减少epoch数，可以先测试5个epoch)
batch_size = 1  # total_batch_size = #GPU x batch_size
num_workers = 0  # workers for pytorch DataLoader (set to 0 on Windows to avoid pickle issues with lambda functions)
pin_memory = True  # whether pin_memory for pytorch DataLoader
print_freq = 50  # frequency to print logs
starting_epoch = 0
max_norm = 0.1  # clip gradient norm

output_dir = None  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = False  # useful for debugging distributed training

# define dataset for train
# coco_path = "/home/rjzy/PycharmProjects/data/SeaDronesSeeOD2/uncompressed"  # /PATH/TO/YOUR/COCODIR
coco_path = '../seadronessee/Downloads/Uncompressed Version/'
# coco_path = '/home/rjzy/Documents/SalienceDETR/Salience-DETR/data/1000minidata'
# coco_path = '/home/rjzy/Documents/SalienceDETR/Salience-DETR/data/coco'
train_transform = presets.detr  # see transforms/presets to choose a transform
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/images/train",
    ann_file=f"{coco_path}/annotations/instances_train.json",
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/images/val",
    ann_file=f"{coco_path}/annotations/instances_val.json",
    transforms=None,  # the eval_transform is integrated in the model
)

# model config to train
# 轻量级配置（针对CPU训练优化）：降低图像尺寸、减少Transformer层数和queries
model_path = "configs/salience_detr/salience_detr_resnet50_480_640_lite.py"
# 原始配置（GPU训练推荐）
# model_path = "configs/salience_detr/salience_detr_resnet50_800_1333.py"
# model_path = "configs/salience_detr/salience_detr_swin_l_800_1333.py"

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune, for example:
# checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-09_38_50
# checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-09_38_50/best_ap.pth
# checkpoints/salience_detr_resnet50_800_1333/train/2024-06-23-09_55_16/best_ap.pth
# 0715 当前最佳
resume_from_checkpoint = "checkpoints/salience_detr_resnet50_800_1333/train/2024-09-11-15_11_51/best_ap.pth"
# resume_from_checkpoint = None
learning_rate = 1e-4  # initial learning rate
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)
