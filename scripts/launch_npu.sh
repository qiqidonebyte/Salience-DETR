#!/bin/bash

# NPU训练启动脚本
# 使用方法: bash scripts/launch_npu.sh --config configs/npu_config.py

set -e

# 默认参数
CONFIG_FILE="configs/npu_config.py"
OUTPUT_DIR=""
RESUME=""
BATCH_SIZE=""
NUM_EPOCHS=24
LEARNING_RATE=""
SEED=42
NPU_IDS=""  # 不设置则使用环境变量
MIXED_PRECISION="bf16"
USE_APEX=true
OPT_LEVEL="O2"
GRADIENT_ACCUMULATION=1
LOG_LEVEL="INFO"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --npu-ids)
            NPU_IDS="$2"
            shift 2
            ;;
        --mixed-precision)
            MIXED_PRECISION="$2"
            shift 2
            ;;
        --no-apex)
            USE_APEX=false
            shift
            ;;
        --opt-level)
            OPT_LEVEL="$2"
            shift 2
            ;;
        --gradient-accumulation)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 设置NPU环境变量
if [ -n "$NPU_IDS" ]; then
    export ASCEND_VISIBLE_DEVICES=$NPU_IDS
    echo "设置ASCEND_VISIBLE_DEVICES=$NPU_IDS"
else
    echo "使用环境变量ASCEND_VISIBLE_DEVICES: ${ASCEND_VISIBLE_DEVICES:-0}"
fi

# 检查NPU数量
NPU_COUNT=$(echo ${ASCEND_VISIBLE_DEVICES:-0} | tr ',' '\n' | wc -l)
echo "检测到 $NPU_COUNT 个NPU设备"

# 设置分布式训练参数
if [ $NPU_COUNT -gt 1 ]; then
    echo "启用分布式训练"
    DISTRIBUTED_ARGS="
        --nproc_per_node $NPU_COUNT
        --master_port 29500
    "
else
    echo "单卡训练"
    DISTRIBUTED_ARGS=""
fi

# 构建训练命令
TRAIN_CMD="python main_npu.py
    --config-file $CONFIG_FILE
    --seed $SEED
    --log-level $LOG_LEVEL
    --mixed-precision $MIXED_PRECISION
    --gradient-accumulation $GRADIENT_ACCUMULATION
    --opt-level $OPT_LEVEL"

# 添加可选参数
if [ -n "$OUTPUT_DIR" ]; then
    TRAIN_CMD="$TRAIN_CMD --output-dir $OUTPUT_DIR"
fi

if [ -n "$RESUME" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
fi

if [ -n "$BATCH_SIZE" ]; then
    TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
fi

if [ -n "$NUM_EPOCHS" ]; then
    TRAIN_CMD="$TRAIN_CMD --num-epochs $NUM_EPOCHS"
fi

if [ -n "$LEARNING_RATE" ]; then
    TRAIN_CMD="$TRAIN_CMD --lr $LEARNING_RATE"
fi

if [ "$USE_APEX" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use-apex"
fi

# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/training_$(date +"%Y%m%d_%H%M%S").log"

echo "开始时间: $(date)"
echo "训练命令: $TRAIN_CMD"
echo "日志文件: $LOG_FILE"

# 执行训练
if [ $NPU_COUNT -gt 1 ]; then
    # 分布式训练
    torchrun $DISTRIBUTED_ARGS $TRAIN_CMD 2>&1 | tee $LOG_FILE
else
    # 单卡训练
    $TRAIN_CMD 2>&1 | tee $LOG_FILE
fi

echo "结束时间: $(date)"
echo "训练完成!"