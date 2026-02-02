#!/bin/bash

# NPU训练启动脚本

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NPU_VISIBLE_DEVICES=0,1,2,3  # 指定使用的NPU设备
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=192.168.100.100  # 根据实际网络设置

# 训练参数
CONFIG_FILE="configs/train_config.py"
MIXED_PRECISION="bf16"  # NPU建议使用bf16
BATCH_SIZE=8
ACCUM_STEPS=1
SEED=42
NUM_EPOCHS=24
OUTPUT_DIR="checkpoints/npu_train_$(date +%Y%m%d_%H%M%S)"

# 分布式训练参数
WORLD_SIZE=4  # NPU数量
RANK=0
DIST_URL="env://"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 启动训练
echo "开始NPU分布式训练..."
echo "使用设备: $NPU_VISIBLE_DEVICES"
echo "批量大小: $BATCH_SIZE"
echo "混合精度: $MIXED_PRECISION"
echo "输出目录: $OUTPUT_DIR"

python -m torch.distributed.launch \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=29500 \
    main.py \
    --config-file $CONFIG_FILE \
    --mixed-precision $MIXED_PRECISION \
    --batch-size $BATCH_SIZE \
    --accumulate-steps $ACCUM_STEPS \
    --seed $SEED \
    --npu \
    --npu-id 0 \
    --world-size $WORLD_SIZE \
    --rank $RANK \
    --dist-url $DIST_URL \
    --use-apex \
    --opt-level O2 \
    --output-dir $OUTPUT_DIR

echo "训练完成！结果保存在: $OUTPUT_DIR"