from typing import Dict, List, Tuple, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from models.bricks.denoising import GenerateCDNQueries
from models.bricks.losses import sigmoid_focal_loss
from models.detectors.base_detector import DNDETRDetector
from models.bricks.fpn2 import FPN
from models.bricks.msrcr_enhanced import MSRCREnhanced


class SalienceCriterion(nn.Module):
    def __init__(
        self,
        limit_range: Tuple = ((-1, 64), (64, 128), (128, 256), (256, 99999)),
        noise_scale: float = 0.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.limit_range = limit_range
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, foreground_mask, targets, feature_strides, image_sizes):
        gt_boxes_list = []
        for t, (img_h, img_w) in zip(targets, image_sizes):
            boxes = t["boxes"]
            boxes = box_ops._box_cxcywh_to_xyxy(boxes)
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            gt_boxes_list.append(boxes * scale_factor)

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
        foreground_mask = torch.cat([e.flatten(-2) for e in foreground_mask], -1)
        foreground_mask = foreground_mask.squeeze(1)

        if True:
            # 计算目标的显著性得分权重
            size_weights = self.compute_size_weights(gt_boxes_list)
            # 动态调整显著性得分
            adjusted_foreground_mask = self.adjust_salience_scores(foreground_mask, size_weights)

        num_pos = torch.sum(mask_targets > 0.5 * self.noise_scale).clamp_(min=1)
        salience_loss = (
            sigmoid_focal_loss(
                foreground_mask,
                mask_targets,
                num_pos,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * adjusted_foreground_mask.shape[1]
        )
        return {"loss_salience": salience_loss}

    def compute_size_weights(self, gt_boxes_list, min_area_threshold=None):
        # 计算每个目标的面积
        areas = torch.cat([(box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) for box in gt_boxes_list])
        areas = areas.float()  # 确保面积是浮点数

        # 动态确定小目标的面积阈值
        # if min_area_threshold is None:
        #     min_area_threshold = torch.quantile(areas, 0.1)  # 使用10%分位数作为阈值

        # 对小目标赋予更高的权重，使用平方根或对数函数来平滑权重
        # 使用平方根函数来减少小目标面积的影响
        size_weights = torch.sqrt(1.0 / (areas + 1e-4))  # 避免除以零

        # 确保权重不会过大
        size_weights = torch.clamp(size_weights, 0.5, 2.0)

        return size_weights
    
    def adjust_salience_scores(self, foreground_mask, size_weights):
        # 将显著性得分转换为概率
        salience_scores = torch.sigmoid(foreground_mask)  # 转换为概率

        # 根据面积权重调整显著性得分
        # 对于小目标，增加其显著性得分；对于大目标，保持显著性得分基本不变
        adjusted_salience_scores = salience_scores * size_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # 确保调整后的显著性得分在合理范围内
        adjusted_salience_scores = torch.clamp(adjusted_salience_scores, 0.01, 0.99)

        return adjusted_salience_scores
    
    def get_pixel_coordinate(self, feature_shape, stride, device):
        height, width = feature_shape
        coord_y, coord_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride[0],
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride[1],
            indexing="ij",
        )
        coord_y = coord_y.reshape(-1)
        coord_x = coord_x.reshape(-1)
        return coord_x, coord_y

    def get_mask_single_level(self, coord_x, coord_y, gt_boxes, level_idx):
        # gt_label: (m,) gt_boxes: (m, 4)
        # coord_x: (h*w, )
        left_border_distance = coord_x[:, None] - gt_boxes[None, :, 0]  # (h*w, m)
        top_border_distance = coord_y[:, None] - gt_boxes[None, :, 1]
        right_border_distance = gt_boxes[None, :, 2] - coord_x[:, None]
        bottom_border_distance = gt_boxes[None, :, 3] - coord_y[:, None]
        border_distances = torch.stack(
            [left_border_distance, top_border_distance, right_border_distance, bottom_border_distance],
            dim=-1,
        )  # [h*w, m, 4]

        # the foreground queries must satisfy two requirements:
        # 1. the quereis located in bounding boxes
        # 2. the distance from queries to the box center match the feature map stride
        min_border_distances = torch.min(border_distances, dim=-1)[0]  # [h*w, m]
        max_border_distances = torch.max(border_distances, dim=-1)[0]
        mask_in_gt_boxes = min_border_distances > 0
        min_limit, max_limit = self.limit_range[level_idx]
        mask_in_level = (max_border_distances > min_limit) & (max_border_distances <= max_limit)
        mask_pos = mask_in_gt_boxes & mask_in_level

        # scale-independent salience confidence
        row_factor = left_border_distance + right_border_distance
        col_factor = top_border_distance + bottom_border_distance
        delta_x = (left_border_distance - right_border_distance) / row_factor
        delta_y = (top_border_distance - bottom_border_distance) / col_factor
        confidence = torch.sqrt(delta_x**2 + delta_y**2) / 2

        confidence_per_box = 1 - confidence
        confidence_per_box[~mask_in_gt_boxes] = 0

        # process positive coordinates
        if confidence_per_box.numel() != 0:
            mask = confidence_per_box.max(-1)[0]
        else:
            mask = torch.zeros(coord_y.shape, device=confidence.device, dtype=confidence.dtype)

        # process negative coordinates
        mask_pos = mask_pos.long().sum(dim=-1) >= 1
        mask[~mask_pos] = 0

        # add noise to add randomness
        mask = (1 - self.noise_scale) * mask + self.noise_scale * torch.rand_like(mask)
        return mask


# PhysAwareDETR has the architecture similar to FocusDETR
class PhysAwareDETR(DNDETRDetector):
    def __init__(
        # model structure
        self,
        backbone: nn.Module,
        neck: nn.Module,
        position_embedding: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        focus_criterion: nn.Module,
        # model parameters
        num_classes: int = 91,
        # 默认900，增加到1500
        num_queries: int = 1600,
        denoising_nums: int = 100,
        # model variants
        aux_loss: bool = True,
        min_size: int = None,
        max_size: int = None,
        # 物理光学增强模块（可选）
        msrcr_enhanced: nn.Module = None,
    ):
        super().__init__(min_size, max_size)
        # define model parameters
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        embed_dim = transformer.embed_dim

        # define model structures
        self.backbone = backbone

        # 初始化FPN模块
        self.fpn = FPN(backbone.num_channels, embed_dim)

        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.denoising_generator = GenerateCDNQueries(
            num_queries=num_queries,
            num_classes=num_classes,
            label_embed_dim=embed_dim,
            denoising_nums=denoising_nums,
            label_noise_prob=0.5,
            box_noise_scale=1.0,
        )
        self.focus_criterion = focus_criterion
        # 物理光学增强模块（可选）
        self.msrcr_enhanced = msrcr_enhanced
        self.num_feature_levels = getattr(self.transformer, "num_feature_levels", None)
        if self.num_feature_levels is None:
            if hasattr(self.neck, "num_channels"):
                self.num_feature_levels = len(self.neck.num_channels)
            elif hasattr(self.backbone, "num_channels"):
                self.num_feature_levels = len(self.backbone.num_channels)
            else:
                self.num_feature_levels = 3  # 回退到常见的3个尺度
        if self.msrcr_enhanced is not None:
            # 双路径特征融合参数
            # alpha: 控制原始特征和增强特征的融合比例
            # 初始化为0.5，让模型学习最优融合比例
            self.fusion_alpha = nn.Parameter(torch.ones(self.num_feature_levels, 1, 1, 1) * 0.5)
            # beta: 控制物理注意力的强度（用于特征调制）
            # 初始化为0.1，避免初期过度影响
            self.physical_gate_strength = nn.Parameter(torch.ones(self.num_feature_levels, 1, 1, 1) * 0.1)

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        attention_map = None
        enhanced_images = None
        
        # 物理光学增强模块（针对极端天气场景）
        # 创新点：双路径设计 - 原始路径 + 增强路径
        if self.msrcr_enhanced is not None:
            # 反归一化到 [0, 1] 范围，MSRCR需要在这个范围工作
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.tensors.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.tensors.device).view(1, 3, 1, 1)
            denormalized = torch.clamp(images.tensors * std + mean, 0.0, 1.0)
            
            # 获取注意力图和增强后的图像
            attention_map, enhanced_images = self.msrcr_enhanced(denormalized, return_enhanced_image=True)
            
            # 将增强后的图像重新归一化，用于backbone提取特征
            enhanced_images_normalized = (enhanced_images - mean) / std

        # 路径1：原始图像特征提取
        multi_level_feats_original = self.backbone(images.tensors)
        multi_level_feats_original = self.fpn(multi_level_feats_original)
        multi_level_feats_original = self.neck(multi_level_feats_original)
        
        # 路径2：增强图像特征提取（仅在启用MSRCR时）
        if enhanced_images is not None:
            multi_level_feats_enhanced = self.backbone(enhanced_images_normalized)
            multi_level_feats_enhanced = self.fpn(multi_level_feats_enhanced)
            multi_level_feats_enhanced = self.neck(multi_level_feats_enhanced)
            
            # 双路径特征融合：自适应融合原始特征和增强特征
            multi_level_feats = self._fuse_dual_path_features(
                multi_level_feats_original, 
                multi_level_feats_enhanced,
                attention_map
            )
        else:
            multi_level_feats = multi_level_feats_original
            # 即使没有增强路径，也可以应用注意力图（如果存在）
            if attention_map is not None:
                multi_level_feats = self._apply_physical_attention(multi_level_feats, attention_map)

        multi_level_masks = []
        multi_level_position_embeddings = []
        for feature in multi_level_feats:
            multi_level_masks.append(F.interpolate(mask[None], size=feature.shape[-2:]).to(torch.bool)[0])
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        if self.training:
            # collect ground truth for denoising generation
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
            noised_results = self.denoising_generator(gt_labels_list, gt_boxes_list)
            noised_label_query = noised_results[0]
            noised_box_query = noised_results[1]
            attn_mask = noised_results[2]
            denoising_groups = noised_results[3]
            max_gt_num_per_image = noised_results[4]
        else:
            noised_label_query = None
            noised_box_query = None
            attn_mask = None
            denoising_groups = None
            max_gt_num_per_image = None

        # feed into transformer
        outputs_class, outputs_coord, enc_class, enc_coord, foreground_mask = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            noised_label_query,
            noised_box_query,
            attn_mask=attn_mask,
        )
        # hack implementation for distributed training
        outputs_class[0] += self.denoising_generator.label_encoder.weight[0, 0] * 0.0

        # denoising postprocessing
        if denoising_groups is not None and max_gt_num_per_image is not None:
            dn_metas = {
                "denoising_groups": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image,
            }
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas)

            # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # compute loss
            loss_dict = self.criterion(output, targets)
            dn_losses = self.compute_dn_loss(dn_metas, targets)
            loss_dict.update(dn_losses)

            # compute focus loss
            feature_stride = [(
                images.tensors.shape[-2] / feature.shape[-2],
                images.tensors.shape[-1] / feature.shape[-1],
            ) for feature in multi_level_feats]
            focus_loss = self.focus_criterion(foreground_mask, targets, feature_stride, images.image_sizes)
            loss_dict.update(focus_loss)

            # loss reweighting
            weight_dict = self.criterion.weight_dict
            loss_dict = dict((k, loss_dict[k] * weight_dict[k]) for k in loss_dict.keys() if k in weight_dict)
            return loss_dict

        detections = self.postprocessor(output, original_image_sizes)
        return detections

    def _fuse_dual_path_features(
        self, 
        features_original: List[Tensor], 
        features_enhanced: List[Tensor],
        attention_map: Tensor
    ) -> List[Tensor]:
        """
        双路径特征融合：自适应融合原始特征和增强特征
        
        创新点：
        1. 双路径设计：原始路径保留正常天气下的特征，增强路径提供极端天气下的鲁棒特征
        2. 自适应融合：根据图像质量和特征质量，动态调整融合比例
        3. 注意力引导：使用物理注意力图指导融合，突出重要区域
        
        理论支撑：
        - 在正常天气下，原始特征更可靠，融合比例应该偏向原始特征
        - 在极端天气下，增强特征更可靠，融合比例应该偏向增强特征
        - 通过可学习的alpha参数，模型可以自动学习最优融合策略
        """
        fused_features = []
        num_levels = len(features_original)
        num_alpha = self.fusion_alpha.shape[0]
        num_strength = self.physical_gate_strength.shape[0]
        
        for level_idx in range(num_levels):
            feat_orig = features_original[level_idx]
            feat_enh = features_enhanced[level_idx]
            
            # 获取该层级的融合参数
            alpha_idx = min(level_idx, num_alpha - 1)
            strength_idx = min(level_idx, num_strength - 1)
            
            # alpha: 控制原始特征和增强特征的融合比例
            # 使用sigmoid确保在[0, 1]范围内，初始值0.5表示等权重融合
            alpha = torch.sigmoid(self.fusion_alpha[alpha_idx])
            
            # 基础融合：alpha * feat_orig + (1-alpha) * feat_enh
            fused = alpha * feat_orig + (1.0 - alpha) * feat_enh
            
            # 使用物理注意力图进一步调制融合后的特征
            if attention_map is not None:
                attn_resized = F.interpolate(
                    attention_map, 
                    size=feat_orig.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )
                # beta: 控制物理注意力的强度
                beta = torch.sigmoid(self.physical_gate_strength[strength_idx] * 5.0)
                
                # 注意力引导的特征增强
                # 在注意力高的区域，增强特征的影响；在注意力低的区域，保持原始特征
                attention_guided = fused * (1.0 + beta * attn_resized)
                fused_features.append(attention_guided)
            else:
                fused_features.append(fused)
        
        return fused_features
    
    def _apply_physical_attention(self, features: List[Tensor], attention_map: Tensor) -> List[Tensor]:
        """
        使用物理增强生成的注意力图对多尺度特征进行条件化调制。
        这是单路径模式下的注意力应用（当没有增强路径时）。
        """
        fused_features = []
        num_strength = self.physical_gate_strength.shape[0]
        for level_idx, feat in enumerate(features):
            attn_resized = F.interpolate(attention_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            strength_idx = min(level_idx, num_strength - 1)
            level_strength = self.physical_gate_strength[strength_idx]
            
            # 使用可学习的门控，让模型决定是否使用物理注意力
            gated_strength = torch.sigmoid(level_strength * 5.0)
            
            # 残差连接，让模型可以选择性地增强特征
            fused_features.append(feat + gated_strength * (feat * attn_resized - feat))
        return fused_features
