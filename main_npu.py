import argparse
import datetime
import os
import pprint
import re
import time
import sys

import torch
import torch_npu
from torch.optim import AdamW

# 设置全局默认的整数类型
torch.set_default_dtype(torch.float32)

# 检查NPU可用性
if torch_npu.npu.is_available():
    print(f"检测到NPU设备: {torch_npu.npu.device_count()} 个")
    for i in range(torch_npu.npu.device_count()):
        print(f"  NPU {i}: {torch_npu.npu.get_device_name(i)}")
else:
    print("未检测到NPU设备，将使用CPU/GPU训练")

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
    parser.add_argument("--config-file", default="configs/npu_config.py")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",  # NPU建议使用bf16
        choices=["no", "fp16", "bf16"],
        help="混合精度训练选项",
    )
    parser.add_argument(
        "--accumulate-steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")

    # NPU相关参数
    parser.add_argument("--npu", action="store_true", default=torch_npu.npu.is_available(), help="使用NPU训练")
    parser.add_argument("--npu-ids", type=str, default=None, help="NPU设备ID，如'0,1,2,3'")

    # 分布式训练参数
    parser.add_argument("--world-size", type=int, default=-1, help="进程总数")
    parser.add_argument("--rank", type=int, default=-1, help="进程排名")
    parser.add_argument("--local-rank", type=int, default=-1, help="本地进程排名")

    args = parser.parse_args()
    return args


def setup_npu_environment(args):
    """设置NPU环境"""
    if not args.npu or not torch_npu.npu.is_available():
        return args

    # 设置NPU设备
    if args.npu_ids:
        os.environ['ASCEND_VISIBLE_DEVICES'] = args.npu_ids
        npu_ids = args.npu_ids.split(',')
    else:
        npu_ids = os.environ.get('ASCEND_VISIBLE_DEVICES', '0').split(',')

    args.npu_ids = npu_ids
    args.num_npus = len(npu_ids) if npu_ids[0] else 1

    # 设置分布式训练
    if args.num_npus > 1 or args.world_size > 1:
        args.distributed = True
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
        import torch.distributed as dist
        dist.init_process_group(
            backend='hccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank
        )

        print(f"初始化分布式训练: rank={args.rank}, world_size={args.world_size}, local_rank={args.local_rank}")
    else:
        args.distributed = False
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1

    # 设置当前设备
    torch_npu.npu.set_device(args.local_rank)
    args.device = torch.device(f"npu:{args.local_rank}")

    return args


def train():
    args = parse_args()
    cfg = Config(args.config_file, partials=("lr_scheduler", "optimizer", "param_dicts"))

    # 设置NPU环境
    if args.npu:
        args = setup_npu_environment(args)
        print(f"使用NPU设备: npu:{args.local_rank}")

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
            import accelerate
            accelerate.utils.wait_for_everyone()
            cfg.output_dir = os.path.join(
                "checkpoints",
                os.path.basename(cfg.model_path).split(".")[0],
                "train",
                datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"),
            )

    # Initialize accelerator
    project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, total_limit=5, automatic_checkpoint_naming=True
    )
    tensorboard_tracker = TensorBoardTracker(run_name="tf_log", logging_dir=cfg.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_parameters)

    # NPU适配：设置混合精度
    if args.npu and args.mixed_precision == "fp16":
        args.mixed_precision = "bf16"  # NPU建议使用bf16

    accelerator = Accelerator(
        log_with=tensorboard_tracker,
        project_config=project_config,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_steps,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[kwargs],
    )
    accelerator.init_trackers("det_train")
    default_setup(args, cfg, accelerator)

    # instantiate dataset
    params = dict(num_workers=cfg.num_workers, collate_fn=collate_fn)

    # NPU适配：pin_memory设置
    if args.npu:
        params.update(dict(pin_memory=False, persistent_workers=True))
    else:
        params.update(dict(pin_memory=cfg.pin_memory, persistent_workers=True))

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        params.update({"worker_init_fn": seed_worker, "generator": fixed_generator()})

    # we use group_based sampler
    group_ids = create_aspect_ratio_groups(cfg.train_dataset, k=3)

    if args.distributed:
        # 分布式采样器
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(cfg.train_dataset, shuffle=True)
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, cfg.batch_size
        )
        train_loader = data.DataLoader(cfg.train_dataset, batch_sampler=train_batch_sampler, **params)

        test_sampler = DistributedSampler(cfg.test_dataset, shuffle=False)
        test_loader = data.DataLoader(cfg.test_dataset, 1, sampler=test_sampler, **params)
    else:
        train_batch_sampler = GroupedBatchSampler(
            data.RandomSampler(cfg.train_dataset), group_ids, cfg.batch_size
        )
        train_loader = data.DataLoader(cfg.train_dataset, batch_sampler=train_batch_sampler, **params)
        test_loader = data.DataLoader(cfg.test_dataset, 1, shuffle=False, **params)

    # instantiate model
    model = Config(cfg.model_path).model

    # NPU适配：使用标准AdamW，不使用Apex
    if args.npu:
        # NPU使用标准AdamW
        param_dicts = cfg.param_dicts(model)
        optimizer = AdamW(
            param_dicts,
            lr=cfg.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        print("NPU训练：使用标准AdamW优化器")
    else:
        # 使用配置文件中的优化器
        optimizer = cfg.optimizer(cfg.param_dicts(model))

    lr_scheduler = cfg.lr_scheduler(optimizer)

    # register dataset class information
    cat_ids = list(range(max(cfg.train_dataset.coco.cats.keys()) + 1))
    classes = tuple(cfg.train_dataset.coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)
    model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))

    # log the configurations
    logger = get_logger(os.path.basename(os.getcwd()) + "." + __name__)

    # prepare for distributed training
    model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, lr_scheduler
    )

    if getattr(cfg, "resume_from_checkpoint", None) is not None:
        if os.path.isdir(str(cfg.resume_from_checkpoint)):
            accelerator.load_state(cfg.resume_from_checkpoint)
            path = os.path.basename(cfg.resume_from_checkpoint)
            cfg.starting_epoch = int(path.split("_")[-1]) + 1
            accelerator.project_configuration.iteration = cfg.starting_epoch
            logger.info(f"resume training of {cfg.output_dir}, from {path}")
        elif os.path.isfile(str(cfg.resume_from_checkpoint)):
            checkpoint = load_checkpoint(cfg.resume_from_checkpoint)
            checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint

            # NPU适配：处理设备映射
            if args.npu:
                # 将检查点从CPU/GPU映射到NPU
                checkpoint = {k.replace('cuda:', 'npu:'): v for k, v in checkpoint.items()}
                checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

            load_state_dict(accelerator.unwrap_model(model), checkpoint)
            model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))
            logger.info(
                f"load pretrained from {cfg.resume_from_checkpoint}, output_dir is {cfg.output_dir}"
            )
        else:
            logger.warn("resume_from_checkpoint is not a path or a file, skip loading")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("model parameters: {}".format(n_params))
        logger.info("optimizer: {}".format(optimizer))
        logger.info("lr_scheduler: {}".format(pprint.pformat(lr_scheduler.state_dict())))

    # save dataset name
    if accelerator.is_main_process:
        label_file = os.path.join(cfg.output_dir, "label_names.txt")
        with open(label_file, "w") as f:
            caid_name = [f"{k} {v['name']}" for k, v in cfg.train_dataset.coco.cats.items()]
            caid_name = "\n".join(caid_name)
            f.write(caid_name)
        logger.info(f"Label names is saved to {label_file}")

        # 保存训练配置
        config_file = os.path.join(cfg.output_dir, "train_config.txt")
        with open(config_file, "w") as f:
            f.write(f"模型: {cfg.model_path}\n")
            f.write(f"开始时间: {datetime.datetime.now()}\n")
            f.write(f"NPU设备: {args.npu_ids if args.npu_ids else os.environ.get('ASCEND_VISIBLE_DEVICES', '0')}\n")
            f.write(f"NPU数量: {args.world_size}\n")
            f.write(f"批量大小: {cfg.batch_size}\n")
            f.write(f"学习率: {cfg.learning_rate}\n")
            f.write(f"混合精度: {args.mixed_precision}\n")
            f.write(f"优化器: {optimizer.__class__.__name__}\n")
            f.write(f"训练轮数: {cfg.num_epochs}\n")

    logger.info("Start training")
    start_time = time.perf_counter()
    highest_checkpoint = HighestCheckpoint(accelerator, model)

    for epoch in range(cfg.starting_epoch, cfg.num_epochs):
        if args.distributed:
            train_loader._loader.batch_sampler.sampler.set_epoch(epoch)

        train_one_epoch_acc(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epoch=epoch,
            print_freq=cfg.print_freq,
            max_grad_norm=cfg.max_norm,
            accelerator=accelerator,
        )
        lr_scheduler.step()

        # save checkpoint
        accelerator.save_state(safe_serialization=False)
        logger.info("Start evaluation")
        coco_evaluator = evaluate_acc(model, test_loader, epoch, accelerator)

        # save best results
        if coco_evaluator and hasattr(coco_evaluator, 'coco_eval'):
            cur_ap, cur_ap50 = coco_evaluator.coco_eval["bbox"].stats[:2]
            highest_checkpoint.update(ap=cur_ap, ap50=cur_ap50)

        # NPU内存清理
        if args.npu and epoch % 5 == 0:
            torch_npu.npu.empty_cache()

    total_time = time.perf_counter() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time: {}".format(total_time))
    accelerator.end_training()


if __name__ == "__main__":
    train()