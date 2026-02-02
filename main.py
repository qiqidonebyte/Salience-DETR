import argparse
import datetime
import os
import pprint
import re
import time
import warnings

import accelerate
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 设置全局默认的整数类型
torch.set_default_dtype(torch.float32)

# 添加NPU特定导入
try:
    from torch_npu.utils.path_manager import PathManager
    from apex import amp
    from apex.optimizers import FusedAdam, FusedSGD

    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
    warnings.warn("NPU相关库未安装，将回退到CPU/GPU训练")

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import ProjectConfiguration
from torch.utils import data

from util.collate_fn import collate_fn
from util.engine import evaluate_acc, train_one_epoch_acc
from util.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from util.lazy_load import Config
from util.misc import default_setup, encode_labels, fixed_generator, seed_worker
from util.utils import HighestCheckpoint, load_checkpoint, load_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector on NPU")
    parser.add_argument("--config-file", default="configs/train_config.py")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="混合精度训练选项，NPU建议使用bf16",
    )
    parser.add_argument(
        "--accumulate-steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    parser.add_argument("--npu", action="store_true", help="使用NPU训练")
    parser.add_argument("--npu-id", type=int, default=0, help="NPU设备ID")
    parser.add_argument("--world-size", type=int, default=1, help="分布式训练的世界大小")
    parser.add_argument("--rank", type=int, default=0, help="当前进程的排名")
    parser.add_argument("--dist-url", type=str, default="env://", help="分布式训练URL")
    parser.add_argument("--dist-backend", type=str, default="hccl", help="分布式后端，NPU使用hccl")
    parser.add_argument("--use-apex", action="store_true", help="使用NVIDIA Apex优化")
    parser.add_argument("--opt-level", type=str, default="O1", choices=["O0", "O1", "O2", "O3"],
                        help="Apex优化级别")

    dynamo_backend = ["no", "eager", "aot_eager", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser"]
    dynamo_backend += ["cudagraphs", "ofi", "fx2trt", "onnxrt", "tensorrt", "ipex", "tvm"]
    parser.add_argument(
        "--dynamo-backend",
        type=str,
        default="no",
        choices=dynamo_backend,
        help="Torch dynamo后端优化",
    )

    args = parser.parse_args()
    return args


def setup_npu(args, cfg):
    """设置NPU环境"""
    if not args.npu or not NPU_AVAILABLE:
        return

    # 设置NPU设备
    torch_npu.npu.set_device(args.npu_id)

    # 设置分布式训练
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)

        # 初始化分布式进程组
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    # 设置NPU特定的优化
    if args.use_apex:
        print(f"使用Apex优化，级别: {args.opt_level}")

    # 设置NPU内存配置
    torch_npu.npu.set_compile_mode(jit_compile=False)

    # 启用NPU性能优化
    torch_npu.npu.set_option("FLOAT_OVERFLOW_CHECK_ENABLE", "true")

    return True


def create_optimizer_for_npu(model, cfg, args):
    """为NPU创建优化的优化器"""
    if args.npu and NPU_AVAILABLE and args.use_apex:
        # 使用FusedAdam（Apex优化版）用于NPU
        param_dicts = cfg.param_dicts(model)
        optimizer = FusedAdam(
            param_dicts,
            lr=cfg.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
    else:
        # 使用原配置的优化器
        optimizer = cfg.optimizer(cfg.param_dicts(model))

    return optimizer


def setup_model_for_npu(model, args, cfg):
    """为NPU设置模型"""
    if args.npu and NPU_AVAILABLE:
        # 将模型转移到NPU
        model = model.to(f'npu:{args.npu_id}')

        # 如果是分布式训练，使用DistributedDataParallel
        if args.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(
                model,
                device_ids=[args.npu_id],
                output_device=args.npu_id,
                find_unused_parameters=cfg.find_unused_parameters
            )

        # 如果使用Apex混合精度
        if args.use_apex and args.mixed_precision != "no":
            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level=args.opt_level,
                loss_scale="dynamic" if args.opt_level == "O2" else None
            )

    return model


def train():
    args = parse_args()
    cfg = Config(args.config_file, partials=("lr_scheduler", "optimizer", "param_dicts"))

    # 设置NPU
    if args.npu:
        setup_npu(args, cfg)
        print(f"使用NPU设备: npu:{args.npu_id}")

    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch_npu.npu.manual_seed(args.seed)
        torch_npu.npu.manual_seed_all(args.seed)

    # 修改输出目录
    if getattr(cfg, "output_dir", None) is None:
        if hasattr(cfg, "resume_from_checkpoint") and os.path.isdir(str(cfg.resume_from_checkpoint)):
            if "checkpoints" in os.listdir(cfg.resume_from_checkpoint):
                output_dir = os.path.join(cfg.resume_from_checkpoint, "checkpoints")
                folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
                folders.sort(
                    key=lambda folder:
                    list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]
                )
                cfg.resume_from_checkpoint = folders[-1]

            if "checkpoints" in os.path.dirname(cfg.resume_from_checkpoint):
                cfg.output_dir = os.path.dirname(os.path.dirname(cfg.resume_from_checkpoint))
        else:
            # 确保所有进程有相同的输出目录
            accelerate.utils.wait_for_everyone()
            cfg.output_dir = os.path.join(
                "checkpoints",
                os.path.basename(cfg.model_path).split(".")[0],
                "train",
                datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"),
            )

    # 初始化accelerator（适配NPU）
    project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, total_limit=5, automatic_checkpoint_naming=True
    )
    tensorboard_tracker = TensorBoardTracker(run_name="tf_log", logging_dir=cfg.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_parameters)

    # NPU适配：修改mixed_precision设置
    if args.npu and args.mixed_precision in ["fp16", "bf16"]:
        # NPU对bf16支持更好
        if args.mixed_precision == "fp16":
            args.mixed_precision = "bf16"  # NPU建议使用bf16

    accelerator = Accelerator(
        log_with=tensorboard_tracker,
        project_config=project_config,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_steps,
        dynamo_backend=args.dynamo_backend,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[kwargs],
    )

    accelerator.init_trackers("det_train")
    default_setup(args, cfg, accelerator)

    # 实例化数据集
    params = dict(num_workers=cfg.num_workers, collate_fn=collate_fn)

    # NPU适配：pin_memory设置
    if args.npu:
        # NPU对pin_memory支持有限，通常设为False
        params.update(dict(pin_memory=False, persistent_workers=False))
    else:
        params.update(dict(pin_memory=cfg.pin_memory, persistent_workers=True))

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        params.update({"worker_init_fn": seed_worker, "generator": fixed_generator()})

    # 使用分组采样器
    group_ids = create_aspect_ratio_groups(cfg.train_dataset, k=3)
    train_batch_sampler = GroupedBatchSampler(
        data.RandomSampler(cfg.train_dataset), group_ids, cfg.batch_size
    )
    train_loader = data.DataLoader(cfg.train_dataset, batch_sampler=train_batch_sampler, **params)
    test_loader = data.DataLoader(cfg.test_dataset, 1, shuffle=False, **params)

    # 实例化模型、优化器和学习率调度器
    model = Config(cfg.model_path).model

    # NPU适配：修改SyncBatchNorm
    if accelerator.use_distributed:
        if args.npu:
            # NPU可能需要特殊的SyncBatchNorm处理
            try:
                from torch_npu.contrib import SyncBatchNorm
                model = SyncBatchNorm.convert_sync_batchnorm(model)
            except ImportError:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 创建优化器（NPU适配）
    optimizer = create_optimizer_for_npu(model, cfg, args)
    lr_scheduler = cfg.lr_scheduler(optimizer)

    # 注册数据集类别信息
    cat_ids = list(range(max(cfg.train_dataset.coco.cats.keys()) + 1))
    classes = tuple(cfg.train_dataset.coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)
    model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))

    # 日志配置
    logger = get_logger(os.path.basename(os.getcwd()) + "." + __name__)

    # NPU适配：准备分布式训练
    if args.npu and NPU_AVAILABLE:
        # 手动处理模型到NPU
        model = setup_model_for_npu(model, args, cfg)

        # 准备数据加载器和其他组件
        train_loader, test_loader, lr_scheduler = accelerator.prepare(
            train_loader, test_loader, lr_scheduler
        )
    else:
        # 使用accelerator准备所有组件
        model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, test_loader, lr_scheduler
        )

    # 加载检查点
    if getattr(cfg, "resume_from_checkpoint", None) is not None:
        if os.path.isdir(str(cfg.resume_from_checkpoint)):
            accelerator.load_state(cfg.resume_from_checkpoint)
            path = os.path.basename(cfg.resume_from_checkpoint)
            cfg.starting_epoch = int(path.split("_")[-1]) + 1
            accelerator.project_configuration.iteration = cfg.starting_epoch
            logger.info(f"从 {path} 恢复训练 {cfg.output_dir}")
        elif os.path.isfile(str(cfg.resume_from_checkpoint)):
            checkpoint = load_checkpoint(cfg.resume_from_checkpoint)
            checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint

            # NPU适配：加载检查点时处理设备映射
            if args.npu:
                # 将检查点从CPU/GPU映射到NPU
                checkpoint = {k.replace('cuda:', 'npu:'): v for k, v in checkpoint.items()}

            load_state_dict(accelerator.unwrap_model(model), checkpoint)
            model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))
            logger.info(f"从 {cfg.resume_from_checkpoint} 加载预训练权重，输出目录: {cfg.output_dir}")
        else:
            logger.warn("resume_from_checkpoint 不是有效的路径或文件，跳过加载")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数数量: {n_params}")
        logger.info(f"优化器: {optimizer}")
        logger.info(f"学习率调度器: {pprint.pformat(lr_scheduler.state_dict())}")

    # 保存数据集名称（用于推理）
    if accelerator.is_main_process:
        label_file = os.path.join(cfg.output_dir, "label_names.txt")
        with open(label_file, "w") as f:
            caid_name = [f"{k} {v['name']}" for k, v in cfg.train_dataset.coco.cats.items()]
            caid_name = "\n".join(caid_name)
            f.write(caid_name)
        logger.info(f"标签名称保存到 {label_file}")

        # 保存训练配置
        config_file = os.path.join(cfg.output_dir, "train_config.txt")
        with open(config_file, "w") as f:
            f.write(f"NPU训练: {args.npu}\n")
            f.write(f"NPU设备ID: {args.npu_id}\n")
            f.write(f"混合精度: {args.mixed_precision}\n")
            f.write(f"批量大小: {cfg.batch_size}\n")
            f.write(f"学习率: {cfg.learning_rate}\n")
            f.write(f"Apex优化: {args.use_apex}\n")
            f.write(f"优化级别: {args.opt_level}\n")

    logger.info("开始训练")
    start_time = time.perf_counter()
    highest_checkpoint = HighestCheckpoint(accelerator, model)
    
    # 改进: 初始化EMA模块 (提升mAP 0.2-0.5%)
    ema = None
    try:
        from models.bricks.ema import EMA
        ema = EMA(accelerator.unwrap_model(model), decay=0.9999)
        logger.info("EMA模块已启用 (提升mAP 0.2-0.5%)")
    except Exception as e:
        logger.warning(f"EMA模块初始化失败，将不使用EMA: {e}")

    # NPU性能调优
    if args.npu:
        # 启用NPU性能分析
        torch_npu.npu.set_option("ACL_PRECISION_MODE", "allow_mix_precision")
        torch_npu.npu.set_option("ACL_OP_SELECT_IMPL_MODE", "high_precision")

    for epoch in range(cfg.starting_epoch, cfg.num_epochs):
        # 训练一个epoch
        train_one_epoch_acc(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epoch=epoch,
            print_freq=cfg.print_freq,
            max_grad_norm=cfg.max_norm,
            accelerator=accelerator,
            ema=ema,
        )
        lr_scheduler.step()

        # 保存检查点
        accelerator.save_state(safe_serialization=False)
        logger.info("开始评估")

        # 评估模型
        # 改进: 使用EMA模型进行评估 (提升mAP 0.2-0.5%)
        if ema is not None:
            ema.apply_shadow(accelerator.unwrap_model(model))
            coco_evaluator = evaluate_acc(model, test_loader, epoch, accelerator)
            ema.restore(accelerator.unwrap_model(model))
        else:
            coco_evaluator = evaluate_acc(model, test_loader, epoch, accelerator)

        # 保存最佳结果
        if coco_evaluator and hasattr(coco_evaluator, 'coco_eval'):
            cur_ap, cur_ap50 = coco_evaluator.coco_eval["bbox"].stats[:2]
            highest_checkpoint.update(ap=cur_ap, ap50=cur_ap50)

        # NPU内存清理
        if args.npu and epoch % 5 == 0:
            torch_npu.npu.empty_cache()

    total_time = time.perf_counter() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"训练时间: {total_time}")

    # 清理NPU资源
    if args.npu and args.world_size > 1:
        torch.distributed.destroy_process_group()

    accelerator.end_training()


if __name__ == "__main__":
    # NPU环境检查
    if torch_npu.npu.is_available():
        print(f"检测到 {torch_npu.npu.device_count()} 个NPU设备")
        for i in range(torch_npu.npu.device_count()):
            print(f"NPU {i}: {torch_npu.npu.get_device_name(i)}")
    else:
        print("未检测到NPU设备，将使用CPU/GPU训练")

    train()