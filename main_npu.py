import argparse
import datetime
import os
import pprint
import re
import time
import warnings
import sys

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.logging import get_logger
    from accelerate.tracking import TensorBoardTracker
    from accelerate.utils import ProjectConfiguration

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    warnings.warn("accelerate库未安装，将使用自定义NPU训练")

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    warnings.warn("apex库未安装，混合精度训练可能受限")

from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim import AdamW

from util.collate_fn import collate_fn
from util.engine import evaluate_acc, train_one_epoch_acc
from util.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from util.lazy_load import Config
from util.misc import default_setup, encode_labels, fixed_generator, seed_worker
from util.utils import HighestCheckpoint, load_checkpoint, load_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="NPU增强训练脚本")
    parser.add_argument("--config-file", default="configs/npu_config.py", help="配置文件路径")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--resume", default=None, help="恢复训练的检查点路径")
    parser.add_argument("--batch-size", type=int, default=None, help="批量大小")
    parser.add_argument("--num-epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # NPU相关参数
    parser.add_argument("--npu", action="store_true", default=True, help="使用NPU训练")
    parser.add_argument("--npu-ids", type=str, default=None, help="NPU设备ID，如'0,1,2,3'")
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use-apex", action="store_true", default=False, help="使用Apex优化")
    parser.add_argument("--opt-level", type=str, default="O1", choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="梯度累积步数")

    # 分布式训练参数
    parser.add_argument("--dist-backend", type=str, default="hccl", help="分布式后端")
    parser.add_argument("--dist-url", type=str, default="env://", help="分布式URL")
    parser.add_argument("--world-size", type=int, default=-1, help="进程总数")
    parser.add_argument("--rank", type=int, default=-1, help="进程排名")
    parser.add_argument("--local-rank", type=int, default=-1, help="本地进程排名")

    # 性能优化参数
    parser.add_argument("--cudnn-benchmark", action="store_true", help="启用cuDNN基准测试")
    parser.add_argument("--deterministic", action="store_true", help="启用确定性算法")
    parser.add_argument("--async-io", action="store_true", default=True, help="启用异步IO")

    args = parser.parse_args()
    return args


def setup_npu_distributed(args):
    """设置NPU分布式训练环境"""
    if not torch_npu.npu.is_available():
        print("NPU不可用，退出训练")
        sys.exit(1)

    # 设置NPU设备
    if args.npu_ids:
        os.environ['ASCEND_VISIBLE_DEVICES'] = args.npu_ids
        npu_ids = args.npu_ids.split(',')
    else:
        npu_ids = os.environ.get('ASCEND_VISIBLE_DEVICES', '0').split(',')

    args.npu_ids = npu_ids
    args.num_npus = len(npu_ids) if npu_ids[0] else 1

    # 检查是否在分布式环境中
    args.distributed = args.world_size > 1 or args.num_npus > 1

    if args.distributed:
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
        if 'RANK' in os.environ:
            args.rank = int(os.environ['RANK'])
        if 'WORLD_SIZE' in os.environ:
            args.world_size = int(os.environ['WORLD_SIZE'])

        if args.world_size == -1:
            args.world_size = args.num_npus

        if args.rank == -1:
            args.rank = int(os.environ.get('RANK', 0))

        if args.local_rank == -1:
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # 初始化分布式进程组
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

        print(f"初始化分布式训练: rank={args.rank}, world_size={args.world_size}, local_rank={args.local_rank}")

    # 设置当前设备
    torch_npu.npu.set_device(args.local_rank)
    args.device = torch.device(f"npu:{args.local_rank}")

    return args


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_dir(cfg, args):
    """创建输出目录"""
    if args.output_dir:
        cfg.output_dir = args.output_dir
    elif getattr(cfg, "output_dir", None) is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(cfg.model_path).split(".")[0]
        cfg.output_dir = os.path.join(
            "checkpoints",
            f"{model_name}_npu",
            timestamp
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    return cfg


def setup_optimizer(model, cfg, args):
    """设置优化器"""
    param_dicts = cfg.param_dicts(model)

    # NPU环境下使用标准AdamW
    optimizer = AdamW(
        param_dicts,
        lr=args.lr or cfg.learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    return optimizer


def setup_model_for_npu(model, args, cfg):
    """为NPU设置模型"""
    # 将模型移到NPU
    model = model.to(args.device)

    # 分布式训练
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=cfg.find_unused_parameters,
            broadcast_buffers=True
        )

    return model


def move_to_device(data, device):
    """将数据移到指定设备，只移动张量类型"""
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def save_npu_checkpoint(state, filename, args):
    """保存NPU检查点"""
    # 将NPU张量转移到CPU
    cpu_state = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and value.is_npu:
            cpu_state[key] = value.cpu()
        else:
            cpu_state[key] = value

    torch.save(cpu_state, filename)
    print(f"检查点保存到: {filename}")


def train():
    args = parse_args()

    # 加载配置
    cfg = Config(args.config_file, partials=("lr_scheduler", "optimizer", "param_dicts"))

    # 覆盖配置参数
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.num_epochs:
        cfg.num_epochs = args.num_epochs
    if args.resume:
        cfg.resume_from_checkpoint = args.resume

    # 设置NPU
    if args.npu:
        args = setup_npu_distributed(args)
        print(f"使用NPU设备: {args.local_rank}, 设备总数: {args.world_size}")

    # 设置随机种子
    setup_seed(args.seed)

    # 创建输出目录
    cfg = create_output_dir(cfg, args)

    # 初始化日志
    logger = get_logger(__name__)
    logger.setLevel(args.log_level)

    if args.rank <= 0:  # 只在主进程记录
        print("=" * 50)
        print(f"开始训练 - 模型: {cfg.model_path}")
        print(f"输出目录: {cfg.output_dir}")
        print(f"NPU数量: {args.world_size}")
        print(f"批量大小: {cfg.batch_size}")
        print(f"训练轮数: {cfg.num_epochs}")
        print(f"混合精度: {args.mixed_precision}")
        print("=" * 50)

    # 设置性能优化
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    # 实例化数据集
    params = dict(
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0
    )

    if args.deterministic:
        params.update({"worker_init_fn": seed_worker, "generator": fixed_generator()})

    # 使用分组采样器
    group_ids = create_aspect_ratio_groups(cfg.train_dataset, k=3)

    if args.distributed:
        # 分布式采样器
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(cfg.train_dataset, shuffle=True)
        test_sampler = DistributedSampler(cfg.test_dataset, shuffle=False)

        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, cfg.batch_size
        )

        train_loader = data.DataLoader(
            cfg.train_dataset, batch_sampler=train_batch_sampler, **params
        )
        test_loader = data.DataLoader(
            cfg.test_dataset, batch_size=1, sampler=test_sampler, **params
        )
    else:
        train_batch_sampler = GroupedBatchSampler(
            data.RandomSampler(cfg.train_dataset), group_ids, cfg.batch_size
        )
        train_loader = data.DataLoader(
            cfg.train_dataset, batch_sampler=train_batch_sampler, **params
        )
        test_loader = data.DataLoader(
            cfg.test_dataset, batch_size=1, shuffle=False, **params
        )

    # 实例化模型
    model = Config(cfg.model_path).model

    # 注册类别信息
    cat_ids = list(range(max(cfg.train_dataset.coco.cats.keys()) + 1))
    classes = tuple(cfg.train_dataset.coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)
    model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))

    # 设置优化器和学习率调度器
    optimizer = setup_optimizer(model, cfg, args)
    lr_scheduler = cfg.lr_scheduler(optimizer)

    # 为NPU设置模型
    model = setup_model_for_npu(model, args, cfg)

    # 加载检查点
    start_epoch = 0
    if cfg.resume_from_checkpoint:
        if os.path.isfile(cfg.resume_from_checkpoint):
            checkpoint = load_checkpoint(cfg.resume_from_checkpoint)

            # 加载模型状态
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            else:
                model_state = checkpoint

            # 处理NPU设备映射
            if args.npu:
                model_state = {k.replace('cuda:', 'npu:'): v for k, v in model_state.items()}
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}

            # 加载状态
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

            if args.rank <= 0:
                print(f"加载检查点: {cfg.resume_from_checkpoint}")
                if missing_keys:
                    print(f"缺失的键: {missing_keys}")
                if unexpected_keys:
                    print(f"意外的键: {unexpected_keys}")

            # 加载优化器和调度器状态
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1

            # 恢复类别信息
            model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))

    # 保存训练配置
    if args.rank <= 0:
        config_file = os.path.join(cfg.output_dir, "train_config.txt")
        with open(config_file, "w") as f:
            f.write(f"模型: {cfg.model_path}\n")
            f.write(f"开始时间: {datetime.datetime.now()}\n")
            f.write(f"NPU设备: {args.npu_ids}\n")
            f.write(f"NPU数量: {args.world_size}\n")
            f.write(f"批量大小: {cfg.batch_size}\n")
            f.write(f"学习率: {args.lr or cfg.learning_rate}\n")
            f.write(f"混合精度: {args.mixed_precision}\n")
            f.write(f"优化器: {optimizer.__class__.__name__}\n")
            f.write(f"训练轮数: {cfg.num_epochs}\n")
            f.write(f"当前轮数: {start_epoch}\n")

        # 保存标签文件
        label_file = os.path.join(cfg.output_dir, "label_names.txt")
        with open(label_file, "w") as f:
            for cat_id, cat_info in cfg.train_dataset.coco.cats.items():
                f.write(f"{cat_id} {cat_info['name']}\n")

    # 训练循环
    highest_checkpoint = HighestCheckpoint(cfg.output_dir, model)
    start_time = time.time()

    for epoch in range(start_epoch, cfg.num_epochs):
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        # 训练一个epoch
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epoch=epoch,
            args=args,
            cfg=cfg,
            logger=logger
        )

        lr_scheduler.step()

        # 评估
        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs - 1:
            eval_results = evaluate(
                model=model,
                data_loader=test_loader,
                epoch=epoch,
                args=args,
                cfg=cfg,
                logger=logger
            )

            # 保存最佳检查点
            if args.rank <= 0 and eval_results:
                cur_ap = eval_results.get('AP', 0)
                cur_ap50 = eval_results.get('AP50', 0)
                highest_checkpoint.update(
                    ap=cur_ap,
                    ap50=cur_ap50,
                    epoch=epoch,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler
                )

        # 定期保存检查点
        if args.rank <= 0 and (epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs - 1):
            checkpoint_path = os.path.join(cfg.output_dir, f"checkpoint_epoch_{epoch}.pth")
            save_npu_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'ap': highest_checkpoint.best_ap,
                'ap50': highest_checkpoint.best_ap50,
            }, checkpoint_path, args)

        # 清空NPU缓存
        if args.npu and epoch % 5 == 0:
            torch_npu.npu.empty_cache()

    # 训练结束
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if args.rank <= 0:
        logger.info(f"训练完成! 总时间: {total_time_str}")
        logger.info(f"最佳AP: {highest_checkpoint.best_ap:.4f} (epoch {highest_checkpoint.best_ap_epoch})")
        logger.info(f"最佳AP50: {highest_checkpoint.best_ap50:.4f} (epoch {highest_checkpoint.best_ap50_epoch})")

        # 保存最终模型
        final_path = os.path.join(cfg.output_dir, "final_model.pth")
        save_npu_checkpoint({
            'model_state_dict': model.state_dict(),
            'config': cfg.model_path,
            'classes': classes,
        }, final_path, args)

    # 清理分布式训练
    if args.distributed:
        dist.destroy_process_group()


def train_one_epoch(model, optimizer, data_loader, epoch, args, cfg, logger):
    """训练一个epoch"""
    model.train()

    for batch_idx, (images, targets) in enumerate(data_loader):
        # 将数据移到NPU
        images = [img.to(args.device, non_blocking=True) for img in images]

        # 修复：只对张量类型的值进行设备转移
        targets = [
            {
                k: v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        # 前向传播
        with torch.cuda.amp.autocast(enabled=args.mixed_precision != "no" and not args.npu):
            # 注意：NPU使用自己的混合精度机制
            loss_dict = model(images, targets)

        # 计算总损失
        losses = sum(loss for loss in loss_dict.values())

        # 反向传播
        optimizer.zero_grad()
        losses.backward()

        # 梯度裁剪
        if cfg.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)

        # 优化器步进
        optimizer.step()

        # 记录日志
        if args.rank <= 0 and batch_idx % cfg.print_freq == 0:
            loss_str = " ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            logger.info(f"Epoch: {epoch}, Batch: {batch_idx}/{len(data_loader)}, Losses: {loss_str}")

        # 释放内存
        del images, targets, loss_dict, losses

    if args.rank <= 0:
        logger.info(f"Epoch {epoch} 训练完成")


def evaluate(model, data_loader, epoch, args, cfg, logger):
    """评估模型"""
    if args.rank > 0:  # 只在主进程评估
        return None

    model.eval()
    logger.info(f"开始评估 Epoch {epoch}")

    # 这里调用你的评估函数
    # 需要根据你的实际评估代码调整
    try:
        from util.engine import evaluate_acc
        evaluator = evaluate_acc(model, data_loader, epoch, None)  # 最后一个参数是accelerator，可以为None

        if evaluator and hasattr(evaluator, 'coco_eval'):
            results = {
                'AP': evaluator.coco_eval['bbox'].stats[0],
                'AP50': evaluator.coco_eval['bbox'].stats[1],
                'AP75': evaluator.coco_eval['bbox'].stats[2],
            }

            logger.info(f"Epoch {epoch} 评估结果: AP={results['AP']:.4f}, AP50={results['AP50']:.4f}")
            return results
    except Exception as e:
        logger.error(f"评估失败: {e}")

    return None


if __name__ == "__main__":
    train()