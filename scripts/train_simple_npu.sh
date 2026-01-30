#!/bin/bash

# 简化的NPU训练脚本
# 使用方法: ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_simple_npu.sh

set -e

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HCCL_WHITELIST_DISABLE=1
export TASK_QUEUE_ENABLE=1

# 获取NPU设备
NPU_DEVICES=${ASCEND_VISIBLE_DEVICES:-0}
NPU_COUNT=$(echo $NPU_DEVICES | tr ',' '\n' | wc -l)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo "NPU训练启动"
echo "时间: $(date)"
echo "NPU设备: $NPU_DEVICES"
echo "NPU数量: $NPU_COUNT"
echo "========================================"

# 创建日志目录
mkdir -p logs
LOG_FILE="logs/training_npu_${TIMESTAMP}.log"

# 设置训练命令
if [ $NPU_COUNT -gt 1 ]; then
    echo "启用分布式训练 (${NPU_COUNT}卡)"

    torchrun \
        --nproc_per_node=$NPU_COUNT \
        --master_port=29500 \
        main_npu.py \
        --config-file configs/npu_config.py \
        --mixed-precision bf16 \
        --batch-size 4 \
        --num-epochs 24 \
        --lr 1e-4 \
        --seed 42 \
        2>&1 | tee $LOG_FILE
else
    echo "单卡训练"

    python main_npu.py \
        --config-file configs/npu_config.py \
        --mixed-precision bf16 \
        --batch-size 4 \
        --num-epochs 24 \
        --lr 1e-4 \
        --seed 42 \
        2>&1 | tee $LOG_FILE
fi

echo "训练完成! 日志文件: $LOG_FILE"