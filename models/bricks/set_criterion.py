import copy
from typing import Dict

import torch
import torch.distributed
from torch import nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from models.bricks.losses import sigmoid_focal_loss, vari_sigmoid_focal_loss
from util.utils import get_world_size, is_dist_avail_and_initialized


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
            self,
            num_classes: int,
            matcher: nn.Module,
            weight_dict: Dict,
            alpha: float = 0.25,
            gamma: float = 2.0,
            two_stage_binary_cls=False,
            # 新增参数：物理光学损失相关
            use_optical_loss: bool = False,
            optical_loss_weight: float = 0.1,
            optical_loss_module: nn.Module = None,
            # 优化参数
            optical_loss_freq: int = 1,  # 每N次迭代计算一次光学损失
            progressive_weight: bool = False,  # 是否使用渐进式权重
    ):
        """Create the criterion.

        :param num_classes: number of object categories, omitting the special no-object category
        :param matcher: module able to compute a matching between targets and proposals
        :param weight_dict: dict containing as key the names of the losses and as values their relative weight
        :param alpha: alpha in Focal Loss, defaults to 0.25
        :param gamma: gamma in Focal loss, defaults to 2.0
        :param two_stage_binary_cls: Whether to use two-stage binary classification loss, defaults to False
        :param use_optical_loss: 是否使用物理光学损失，默认为False
        :param optical_loss_weight: 物理光学损失的权重
        :param optical_loss_module: 物理光学损失模块实例
        :param optical_loss_freq: 光学损失计算频率（每N次迭代计算一次）
        :param progressive_weight: 是否使用渐进式权重（随训练轮数增加）
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.two_stage_binary_cls = two_stage_binary_cls

        # 物理光学损失相关初始化
        self.use_optical_loss = use_optical_loss
        self.optical_loss_weight = optical_loss_weight
        self.optical_loss_module = optical_loss_module
        self.optical_loss_freq = optical_loss_freq
        self.progressive_weight = progressive_weight

        # 确保weight_dict中包含光学损失权重
        if self.use_optical_loss and "loss_optical" not in self.weight_dict:
            self.weight_dict["loss_optical"] = self.optical_loss_weight

        # 训练状态跟踪
        self.current_iteration = 0
        self.current_epoch = 0

    def loss_labels(self, outputs, targets, num_boxes, indices, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                ) * src_logits.shape[1]
        )
        losses = {"loss_class": loss_class}
        return losses

    def loss_boxes(self, outputs, targets, num_boxes, indices, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops._box_cxcywh_to_xyxy(src_boxes),
                box_ops._box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_optical(self, outputs, targets, num_boxes, indices, **kwargs):
        """计算物理光学损失

        该损失基于FPN特征和物理光学约束，用于增强模型在水下环境中的鲁棒性。
        需要确保outputs中包含'fpn_features'键。
        """
        if not self.use_optical_loss or self.optical_loss_module is None:
            return {"loss_optical": torch.tensor(0.0, device=next(iter(outputs.values())).device)}

        # 检查是否应该计算光学损失（基于频率）
        if self.optical_loss_freq > 1 and self.current_iteration % self.optical_loss_freq != 0:
            return {"loss_optical": torch.tensor(0.0, device=next(iter(outputs.values())).device)}

        # 确保有FPN特征
        if "fpn_features" not in outputs:
            # 如果没有FPN特征，尝试从其他键获取
            if "features" in outputs:
                fpn_features = outputs["features"]
            elif "backbone_features" in outputs:
                fpn_features = outputs["backbone_features"]
            else:
                # 如果都没有，返回零损失
                return {"loss_optical": torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        else:
            fpn_features = outputs["fpn_features"]

        # 检查是否有深度信息（可选）
        depth_maps = None
        if "depth_maps" in outputs:
            depth_maps = outputs["depth_maps"]
        elif "depth" in outputs:
            depth_maps = outputs["depth"]

        # 计算光学损失
        try:
            optical_loss = self.optical_loss_module(fpn_features, depth_maps)

            # 应用渐进式权重（如果启用）
            if self.progressive_weight:
                # 渐进式权重：随训练轮数线性增加，从0到optical_loss_weight
                max_epochs = kwargs.get("max_epochs", 100)
                progress = min(self.current_epoch / max_epochs, 1.0)
                current_weight = self.optical_loss_weight * progress
                optical_loss = optical_loss * current_weight
            else:
                optical_loss = optical_loss * self.optical_loss_weight

        except Exception as e:
            print(f"Warning: Optical loss computation failed: {e}")
            optical_loss = torch.tensor(0.0, device=fpn_features.device)

        return {"loss_optical": optical_loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def calculate_loss(self, outputs, targets, num_boxes, indices=None, **kwargs):
        losses = {}
        # get matching results for each image
        if not indices:
            gt_boxes, gt_labels = list(zip(*map(lambda x: (x["boxes"], x["labels"]), targets)))
            pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
            indices = list(map(self.matcher, pred_boxes, pred_logits, gt_boxes, gt_labels))

        # 计算基础损失
        loss_class = self.loss_labels(outputs, targets, num_boxes, indices=indices)
        loss_boxes = self.loss_boxes(outputs, targets, num_boxes, indices=indices)

        # 计算物理光学损失（如果启用）
        loss_optical = self.loss_optical(outputs, targets, num_boxes, indices=indices, **kwargs)

        losses.update(loss_class)
        losses.update(loss_boxes)
        losses.update(loss_optical)

        return losses

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation

        :param outputs: dict of tensors, see the output specification of the model for the format
        :param targets: list of dicts, such that len(targets) == batch_size
        :param kwargs: 可选参数，如当前epoch、max_epochs等
        :return: a dict containing losses
        """
        # 更新训练状态
        if "current_epoch" in kwargs:
            self.current_epoch = kwargs["current_epoch"]
        if "current_iteration" in kwargs:
            self.current_iteration = kwargs["current_iteration"]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            data=[num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        matching_outputs = {k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"}
        losses.update(self.calculate_loss(matching_outputs, targets, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # get matching results for each image
                losses_aux = self.calculate_loss(aux_outputs, targets, num_boxes, **kwargs)
                losses.update({k + f"_{i}": v for k, v in losses_aux.items()})

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            if self.two_stage_binary_cls:
                for bt in bin_targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            losses_enc = self.calculate_loss(enc_outputs, bin_targets, num_boxes, **kwargs)
            losses.update({k + f"_enc": v for k, v in losses_enc.items()})

        # 更新迭代计数器
        self.current_iteration += 1

        return losses


class HybridSetCriterion(SetCriterion):
    def loss_labels(self, outputs, targets, num_boxes, indices, **kwargs):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou_score = torch.diag(
            box_ops.box_iou(
                box_ops._box_cxcywh_to_xyxy(src_boxes),
                box_ops._box_cxcywh_to_xyxy(target_boxes),
            )
        ).detach()  # add detach according to RT-DETR

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        # construct onehot targets, shape: (batch_size, num_queries, num_classes)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]

        # construct iou_score, shape: (batch_size, num_queries)
        target_score = torch.zeros_like(target_classes, dtype=iou_score.dtype)
        target_score[idx] = iou_score

        loss_class = (
                vari_sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    target_score,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                ) * src_logits.shape[1]
        )
        losses = {"loss_class": loss_class}
        return losses