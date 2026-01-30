#!/bin/bash

# 与原命令完全兼容的NPU训练脚本
# 使用方法: ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash scripts/train_npu_original.sh > ./logs/training$(date +"%Y%m%d_%H%M").log 2>&1 &

# 设置工作目录
cd /app/Salience-DETR

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 设置NPU环境变量
export HCCL_WHITELIST_DISABLE=1
export TASK_QUEUE_ENABLE=1

# 获取NPU设备数量
NPU_DEVICES=${ASCEND_VISIBLE_DEVICES:-0}
NPU_ARRAY=(${NPU_DEVICES//,/ })
NPU_COUNT=${#NPU_ARRAY[@]}

echo "========================================"
echo "Salience-DETR NPU训练启动"
echo "启动时间: $(date)"
echo "工作目录: $(pwd)"
echo "NPU设备: $NPU_DEVICES"
echo "NPU数量: $NPU_COUNT"
echo "========================================"

# 训练参数
BATCH_SIZE=4
LEARNING_RATE=1e-4
NUM_EPOCHS=24
SEED=42
MODEL_CONFIG="configs/salience_detr/salience_detr_resnet50_800_1333.py"

echo "训练配置:"
echo "  模型配置: $MODEL_CONFIG"
echo "  批量大小: $BATCH_SIZE"
echo "  学习率: $LEARNING_RATE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  随机种子: $SEED"
echo "  混合精度: bf16"
echo "========================================"

# 执行训练
if [ $NPU_COUNT -gt 1 ]; then
    echo "启动分布式训练 (${NPU_COUNT}卡)..."

    # 使用torchrun启动分布式训练
    torchrun \
        --nproc_per_node=$NPU_COUNT \
        --master_port=29500 \
        main_npu.py \
        --config-file configs/npu_config.py \
        --mixed-precision bf16 \
        --batch-size $BATCH_SIZE \
        --num-epochs $NUM_EPOCHS \
        --lr $LEARNING_RATE \
        --seed $SEED
else
    echo "启动单卡训练..."

    # 单卡训练
    python main_npu.py \
        --config-file configs/npu_config.py \
        --mixed-precision bf16 \
        --batch-size $BATCH_SIZE \
        --num-epochs $NUM_EPOCHS \
        --lr $LEARNING_RATE \
        --seed $SEED
fi

echo "训练结束时间: $(date)"
echo "========================================"