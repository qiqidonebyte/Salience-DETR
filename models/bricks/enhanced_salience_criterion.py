import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
from typing import Dict, List, Tuple, Any, Optional
import math

from models.bricks.losses import sigmoid_focal_loss


class EnhancedSalienceCriterion(nn.Module):
    """增强的显著性损失，专门针对海洋小目标"""

    def __init__(
            self,
            limit_range: Tuple = ((-1, 64), (64, 128), (128, 256), (256, 99999)),
            noise_scale: float = 0.0,
            alpha: float = 0.25,
            gamma: float = 2.0,
            small_target_weight: float = 2.0,
            marine_specific: bool = True,
            wave_pattern_weight: float = 1.2,
            texture_weight: float = 1.5,
    ):
        super().__init__()
        self.limit_range = limit_range
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.gamma = gamma
        self.small_target_weight = small_target_weight
        self.marine_specific = marine_specific
        self.wave_pattern_weight = wave_pattern_weight
        self.texture_weight = texture_weight

    def forward(self, foreground_mask, targets, feature_strides, image_sizes):
        """计算增强的显著性损失"""
        # 获取GT框
        gt_boxes_list = []
        for t, (img_h, img_w) in zip(targets, image_sizes):
            boxes = t["boxes"]
            boxes = box_ops._box_cxcywh_to_xyxy(boxes)
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            gt_boxes_list.append(boxes * scale_factor)

        # 生成mask目标
        mask_targets = []
        for level_idx, (mask, feature_stride) in enumerate(zip(foreground_mask, feature_strides)):
            feature_shape = mask.shape[-2:]
            coord_x, coord_y = self.get_pixel_coordinate(feature_shape, feature_stride, device=mask.device)
            masks_per_level = []
            for gt_boxes in gt_boxes_list:
                mask = self.get_mask_single_level(coord_x, coord_y, gt_boxes, level_idx)
                masks_per_level.append(mask)
            masks_per_level = torch.stack(masks_per_level)
            mask_targets.append(masks_per_level)

        mask_targets = torch.cat(mask_targets, dim=1)
        foreground_mask_flat = torch.cat([e.flatten(-2) for e in foreground_mask], -1)
        foreground_mask_flat = foreground_mask_flat.squeeze(1)

        # 计算小目标权重
        size_weights = self.compute_size_weights(gt_boxes_list)

        # 动态调整显著性得分
        adjusted_foreground_mask = self.adjust_salience_scores(foreground_mask_flat, size_weights)

        # 计算基础显著性损失
        num_pos = torch.sum(mask_targets > 0.5 * self.noise_scale).clamp_(min=1)
        salience_loss = sigmoid_focal_loss(
            foreground_mask_flat,
            mask_targets,
            num_pos,
            alpha=self.alpha,
            gamma=self.gamma,
        ) * adjusted_foreground_mask.shape[1]

        loss_dict = {"loss_salience": salience_loss}

        # 计算小目标专项损失
        small_target_loss = self.compute_marine_small_target_loss(
            foreground_mask, targets, feature_strides, image_sizes
        )
        loss_dict['loss_small_target'] = small_target_loss

        # 计算海洋特定损失
        if self.marine_specific:
            marine_loss = self.compute_marine_specific_loss(foreground_mask, targets, image_sizes)
            loss_dict.update(marine_loss)

        return loss_dict

    def compute_size_weights(self, gt_boxes_list, min_area_threshold=None):
        """计算基于目标大小的权重"""
        areas = torch.cat([(box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) for box in gt_boxes_list])
        areas = areas.float()

        # 使用平方根函数平滑权重
        size_weights = torch.sqrt(1.0 / (areas + 1e-4))
        size_weights = size_weights / size_weights.mean()  # 归一化
        size_weights = torch.clamp(size_weights, 0.5, 3.0)  # 限制范围

        return size_weights

    def adjust_salience_scores(self, foreground_mask, size_weights):
        """根据目标大小调整显著性得分"""
        salience_scores = torch.sigmoid(foreground_mask)

        # 为每个像素分配权重
        B, H, W = salience_scores.shape
        if len(size_weights) > 0:
            # 扩展权重到每个像素
            weight_map = torch.ones_like(salience_scores)
            adjusted_salience_scores = salience_scores * weight_map
        else:
            adjusted_salience_scores = salience_scores

        adjusted_salience_scores = torch.clamp(adjusted_salience_scores, 0.01, 0.99)
        return adjusted_salience_scores

    def compute_marine_small_target_loss(
            self,
            foreground_mask: List[Tensor],
            targets: List[Dict],
            feature_strides: List[Tuple[int, int]],
            image_sizes: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """海洋小目标专项损失"""
        all_losses = []

        for i, (mask_per_level, target) in enumerate(zip(zip(*foreground_mask), targets)):
            # 计算目标面积
            boxes = target["boxes"]
            img_h, img_w = image_sizes[i]

            # 归一化坐标转换
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            boxes_pixel = box_ops._box_cxcywh_to_xyxy(boxes) * scale_factor.unsqueeze(0)

            # 计算像素面积
            areas = (boxes_pixel[:, 2] - boxes_pixel[:, 0]) * (boxes_pixel[:, 3] - boxes_pixel[:, 1])
            small_indices = areas < 0.002 * (img_h * img_w)  # 面积小于0.2%的视为小目标

            if not small_indices.any():
                continue

            # 为小目标计算损失
            for level_idx, (mask, stride) in enumerate(zip(mask_per_level, feature_strides)):
                B, C, H, W = mask.shape

                # 为小目标生成GT mask
                small_mask_target = self.generate_small_target_mask(
                    mask, boxes_pixel[small_indices], stride, img_h, img_w
                )

                if small_mask_target is None or small_mask_target.sum() == 0:
                    continue

                # 计算focal loss
                mask_flat = mask.flatten(-2)
                loss = sigmoid_focal_loss(
                    mask_flat,
                    small_mask_target,
                    alpha=self.alpha * 1.5,
                    gamma=self.gamma + 0.5,
                )

                all_losses.append(loss * self.small_target_weight)

        if len(all_losses) == 0:
            return torch.tensor(0.0, device=foreground_mask[0].device)

        return torch.stack(all_losses).mean()

    def compute_marine_specific_loss(
            self,
            foreground_mask: List[Tensor],
            targets: List[Dict],
            image_sizes: List[Tuple[int, int]]
    ) -> Dict[str, torch.Tensor]:
        """海洋特定损失"""
        losses = {}

        # 纹理一致性损失
        texture_loss = self.texture_consistency_loss(foreground_mask)
        if texture_loss is not None:
            losses['loss_texture'] = texture_loss * 0.3 * self.texture_weight

        # 边缘清晰度损失
        edge_loss = self.edge_sharpness_loss(foreground_mask)
        if edge_loss is not None:
            losses['loss_edge'] = edge_loss * 0.2

        # 海浪模式损失
        wave_loss = self.wave_pattern_loss(foreground_mask, targets, image_sizes)
        if wave_loss is not None:
            losses['loss_wave'] = wave_loss * 0.1 * self.wave_pattern_weight

        return losses

    def texture_consistency_loss(self, masks: List[Tensor]) -> Optional[torch.Tensor]:
        """纹理一致性损失"""
        if len(masks) < 2:
            return None

        losses = []
        for mask in masks:
            B, _, H, W = mask.shape
            if H <= 2 or W <= 2:
                continue

            # 计算局部纹理方差
            kernel = torch.ones(1, 1, 3, 3, device=mask.device) / 9.0
            local_mean = F.conv2d(mask, kernel, padding=1)
            local_var = (mask - local_mean).pow(2).mean()

            # 我们希望小目标区域内部纹理一致
            # 但不要过于平滑，所以加入正则化
            smoothness_penalty = torch.exp(-local_var * 10)
            loss = local_var + 0.1 * smoothness_penalty
            losses.append(loss)

        return torch.stack(losses).mean() if losses else None

    def edge_sharpness_loss(self, masks: List[Tensor]) -> Optional[torch.Tensor]:
        """边缘清晰度损失"""
        if not masks:
            return None

        losses = []
        for mask in masks:
            B, C, H, W = mask.shape
            if H <= 2 or W <= 2:
                continue

            # 使用Sobel算子计算边缘
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   device=mask.device).view(1, 1, 3, 3).float()
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   device=mask.device).view(1, 1, 3, 3).float()

            edges_x = F.conv2d(mask, sobel_x, padding=1)
            edges_y = F.conv2d(mask, sobel_y, padding=1)
            edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2) + 1e-6)

            # 边缘应该有适中的强度
            target_edge_strength = 0.3
            edge_loss = F.mse_loss(edges, torch.ones_like(edges) * target_edge_strength)
            losses.append(edge_loss)

        return torch.stack(losses).mean() if losses else None

    def wave_pattern_loss(self, masks: List[Tensor], targets: List[Dict],
                          image_sizes: List[Tuple[int, int]]) -> Optional[torch.Tensor]:
        """海浪模式损失（简化版）"""
        if not masks or not targets:
            return None

        # 计算目标的相对位置模式
        all_boxes = []
        for target, (img_h, img_w) in zip(targets, image_sizes):
            boxes = target["boxes"]
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            boxes_pixel = box_ops._box_cxcywh_to_xyxy(boxes) * scale_factor.unsqueeze(0)
            all_boxes.append(boxes_pixel)

        # 如果目标太少，返回None
        if sum(len(b) for b in all_boxes) < 2:
            return None

        # 计算目标间的距离分布
        distances = []
        for boxes in all_boxes:
            if len(boxes) > 1:
                # 计算中心点
                centers = torch.stack([
                    (boxes[:, 0] + boxes[:, 2]) / 2,
                    (boxes[:, 1] + boxes[:, 3]) / 2
                ], dim=1)

                # 计算成对距离
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = torch.norm(centers[i] - centers[j])
                        distances.append(dist)

        if len(distances) == 0:
            return None

        distances = torch.stack(distances)

        # 计算距离分布的熵（我们希望距离分布有一定规律）
        hist = torch.histc(distances, bins=10, min=0, max=distances.max())
        hist = hist / hist.sum() + 1e-6
        entropy = -(hist * torch.log(hist)).sum()

        # 适中的熵值最好（既不要太集中也不要太分散）
        target_entropy = math.log(5)  # 经验值
        loss = F.mse_loss(entropy.unsqueeze(0), torch.tensor([target_entropy], device=entropy.device))

        return loss

    def generate_small_target_mask(self, mask: torch.Tensor, small_boxes: torch.Tensor,
                                   stride: Tuple[int, int], img_h: int, img_w: int) -> Optional[torch.Tensor]:
        """为小目标生成mask"""
        B, C, H, W = mask.shape
        if len(small_boxes) == 0:
            return None

        # 生成坐标网格
        coord_x, coord_y = self.get_pixel_coordinate((H, W), stride, device=mask.device)

        # 初始化mask
        mask_target = torch.zeros(H * W, device=mask.device)

        for box in small_boxes:
            x1, y1, x2, y2 = box

            # 计算每个像素点到框边界的距离
            left_dist = coord_x - x1
            right_dist = x2 - coord_x
            top_dist = coord_y - y1
            bottom_dist = y2 - coord_y

            # 在框内的像素
            inside_box = (left_dist > 0) & (right_dist > 0) & (top_dist > 0) & (bottom_dist > 0)

            if inside_box.any():
                # 计算中心距离权重
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                dist_to_center = torch.sqrt((coord_x - center_x) ** 2 + (coord_y - center_y) ** 2)

                # 距离越近权重越高
                max_dist = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2
                weight = 1.0 - dist_to_center / (max_dist + 1e-6)
                weight = torch.clamp(weight, 0, 1)

                mask_target[inside_box] = torch.maximum(mask_target[inside_box], weight[inside_box])

        return mask_target.view(1, 1, H, W)

    def get_pixel_coordinate(self, feature_shape, stride, device):
        """获取像素坐标"""
        height, width = feature_shape
        stride_h, stride_w = stride

        coord_y, coord_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride_h,
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride_w,
            indexing="ij",
        )
        coord_y = coord_y.reshape(-1)
        coord_x = coord_x.reshape(-1)
        return coord_x, coord_y

    def get_mask_single_level(self, coord_x, coord_y, gt_boxes, level_idx):
        """为单个特征层生成mask"""
        if len(gt_boxes) == 0:
            return torch.zeros(coord_x.shape[0], device=coord_x.device)

        # 计算像素点到框边界的距离
        left_border_distance = coord_x[:, None] - gt_boxes[None, :, 0]
        top_border_distance = coord_y[:, None] - gt_boxes[None, :, 1]
        right_border_distance = gt_boxes[None, :, 2] - coord_x[:, None]
        bottom_border_distance = gt_boxes[None, :, 3] - coord_y[:, None]

        border_distances = torch.stack(
            [left_border_distance, top_border_distance, right_border_distance, bottom_border_distance],
            dim=-1,
        )

        # 判断像素是否在框内
        min_border_distances = torch.min(border_distances, dim=-1)[0]
        max_border_distances = torch.max(border_distances, dim=-1)[0]
        mask_in_gt_boxes = min_border_distances > 0

        # 判断是否在当前特征层
        min_limit, max_limit = self.limit_range[level_idx]
        mask_in_level = (max_border_distances > min_limit) & (max_border_distances <= max_limit)
        mask_pos = mask_in_gt_boxes & mask_in_level

        # 计算置信度
        row_factor = left_border_distance + right_border_distance
        col_factor = top_border_distance + bottom_border_distance

        # 避免除零
        row_factor = torch.where(row_factor == 0, torch.ones_like(row_factor), row_factor)
        col_factor = torch.where(col_factor == 0, torch.ones_like(col_factor), col_factor)

        delta_x = (left_border_distance - right_border_distance) / row_factor
        delta_y = (top_border_distance - bottom_border_distance) / col_factor
        confidence = torch.sqrt(delta_x ** 2 + delta_y ** 2) / 2

        confidence_per_box = 1 - confidence
        confidence_per_box[~mask_in_gt_boxes] = 0

        # 处理正样本
        if confidence_per_box.numel() != 0:
            mask = confidence_per_box.max(-1)[0]
        else:
            mask = torch.zeros(coord_y.shape[0], device=coord_x.device, dtype=coord_x.dtype)

        # 处理负样本
        mask_pos = mask_pos.long().sum(dim=-1) >= 1
        mask[~mask_pos] = 0

        # 添加噪声
        if self.training and self.noise_scale > 0:
            mask = (1 - self.noise_scale) * mask + self.noise_scale * torch.rand_like(mask)

        return mask