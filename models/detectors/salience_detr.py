from typing import Dict, List, Tuple, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from models.bricks.denoising import GenerateCDNQueries
from models.bricks.losses import sigmoid_focal_loss
from models.detectors.base_detector import DNDETRDetector
from models.bricks.fpn2 import OpticalPhysicsFPN as FPN
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
class SalienceDETR(DNDETRDetector):
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
            # 物理感知注意力门控参数（每个特征层独立）
            # 初始化为较小值，让模型逐步学习如何利用物理注意力
            # 使用可学习参数让模型自适应决定每层的注意力强度
            self.physical_gate_strength = nn.Parameter(torch.zeros(self.num_feature_levels, 1, 1, 1))

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        # ================================================================
        # 物理感知引导注意力机制 (Physics-Aware Guided Attention)
        # ================================================================
        # 设计：轻量级并行旁路
        #   - 主路径：原始图像 → Backbone → FPN → Neck → 特征
        #   - 旁路：  原始图像 → MSRCR模块 → 物理感知注意力图
        #   - 融合：  注意力图引导主路径特征，告诉网络"关注哪些区域"
        # ================================================================
        
        physical_attention_map = None
        
        # 旁路：物理感知注意力生成（轻量级，与Backbone并行）
        if self.msrcr_enhanced is not None:
            # 反归一化到 [0, 1] 范围
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.tensors.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.tensors.device).view(1, 3, 1, 1)
            denormalized = torch.clamp(images.tensors * std + mean, 0.0, 1.0)
            
            # 生成物理感知注意力图
            # 注意力图含义：高值区域表示受天气影响大，需要更多关注
            physical_attention_map = self.msrcr_enhanced(denormalized)

        # 主路径：原始图像特征提取
        multi_level_feats = self.backbone(images.tensors)
        multi_level_feats = self.fpn(multi_level_feats)
        multi_level_feats = self.neck(multi_level_feats)
        
        # 融合：用物理感知注意力图引导主路径特征
        if physical_attention_map is not None:
            multi_level_feats = self._apply_physical_attention(multi_level_feats, physical_attention_map)

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

    def _apply_physical_attention(self, features: List[Tensor], attention_map: Tensor) -> List[Tensor]:
        """
        物理感知注意力引导的特征调制
        
        设计思想：
            注意力图告诉主网络"应该更关注原始图像的哪些区域"（如被薄雾遮挡的区域）
            通过门控机制，模型可以自适应地决定在每个特征层如何利用物理注意力
        
        公式：
            output = feat * (1 + gate * attention)
            - feat: 主路径特征
            - attention: 物理感知注意力图（高值=需要更多关注）
            - gate: 可学习的门控参数（控制注意力强度）
        
        理论支撑：
            在极端天气下，某些区域（如被雾遮挡）的特征可能较弱，
            通过物理注意力图增强这些区域的特征响应，提升检测能力
        """
        modulated_features = []
        num_strength = self.physical_gate_strength.shape[0]
        
        for level_idx, feat in enumerate(features):
            # 将注意力图调整到当前特征图尺寸
            attn_resized = F.interpolate(
                attention_map, 
                size=feat.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            # 获取该层的门控强度
            strength_idx = min(level_idx, num_strength - 1)
            gate = self.physical_gate_strength[strength_idx]
            
            # 门控机制：sigmoid确保在[0, 1]范围内
            # 初始化为0时，sigmoid(0)=0.5，模型从适中的注意力强度开始学习
            gated_strength = torch.sigmoid(gate)
            
            # 注意力引导的特征调制
            # 高注意力区域的特征被增强，低注意力区域保持不变
            # 使用残差连接确保训练稳定性
            modulated = feat * (1.0 + gated_strength * attn_resized)
            modulated_features.append(modulated)
        
        return modulated_features
