#!/bin/bash

# 与原命令类似的启动脚本
# 使用方法: ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash scripts/run_npu.sh > ./logs/training_$(date +"%Y%m%d_%H%M").log 2>&1 &

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HCCL_WHITELIST_DISABLE=1
export TASK_QUEUE_ENABLE=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 获取NPU设备
NPU_DEVICES=${ASCEND_VISIBLE_DEVICES:-0}
NPU_COUNT=$(echo $NPU_DEVICES | tr ',' '\n' | wc -l)
echo "使用NPU设备: $NPU_DEVICES (数量: $NPU_COUNT)"

# 设置进程数
WORLD_SIZE=$NPU_COUNT

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_$TIMESTAMP.log"
CONFIG_FILE="configs/npu_config.py"

# 创建日志目录
mkdir -p logs
mkdir -p checkpoints

echo "========================================"
echo "开始NPU训练 - 时间: $(date)"
echo "配置文件: $CONFIG_FILE"
echo "NPU设备: $NPU_DEVICES"
echo "日志文件: $LOG_FILE"
echo "========================================"

# 训练命令
if [ $NPU_COUNT -gt 1 ]; then
    # 多卡分布式训练
    torchrun \
        --nproc_per_node=$NPU_COUNT \
        --master_port=29500 \
        main_npu.py \
        --config-file $CONFIG_FILE \
        --mixed-precision bf16 \
        --use-apex \
        --opt-level O2 \
        --batch-size 4 \
        --num-epochs 24 \
        --lr 1e-4 \
        --seed 42 \
        2>&1 | tee $LOG_FILE
else
    # 单卡训练
    python main_npu.py \
        --config-file $CONFIG_FILE \
        --mixed-precision bf16 \
        --use-apex \
        --opt-level O2 \
        --batch-size 4 \
        --num-epochs 24 \
        --lr 1e-4 \
        --seed 42 \
        2>&1 | tee $LOG_FILE
fi

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "训练成功完成!"
    echo "日志文件: $LOG_FILE"
else
    echo "训练失败，请检查日志: $LOG_FILE"
    exit 1
fi