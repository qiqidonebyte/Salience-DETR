from typing import Dict, List, Tuple, Any, Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
import copy

from models.bricks.denoising import GenerateCDNQueries
from models.bricks.losses import sigmoid_focal_loss
from models.detectors.base_detector import DNDETRDetector
from models.bricks.marine_enhanced_fpn3 import MarineEnhancedFPN_v3
from models.bricks.enhanced_salience_criterion import EnhancedSalienceCriterion
from models.bricks.multi_scale_context_enhancement import (
    MultiScaleContextEnhancement,
    MarineFeatureEnhancement
)


class EnhancedSalienceDETR(DNDETRDetector):
    """增强的SalienceDETR，专门针对海洋小目标检测"""

    def __init__(
            self,
            backbone: nn.Module,
            neck: nn.Module,
            position_embedding: nn.Module,
            transformer: nn.Module,
            criterion: nn.Module,
            postprocessor: nn.Module,
            focus_criterion: nn.Module,
            num_classes: int = 91,
            num_queries: int = 1600,
            denoising_nums: int = 100,
            aux_loss: bool = True,
            min_size: int = None,
            max_size: int = None,
            use_context_enhance: bool = True,
            use_small_target_head: bool = True,
            use_marine_enhance: bool = True,
    ):
        super().__init__(min_size, max_size)
        # 基础参数
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.use_context_enhance = use_context_enhance
        self.use_small_target_head = use_small_target_head
        self.use_marine_enhance = use_marine_enhance

        embed_dim = transformer.embed_dim

        # 基础结构
        self.backbone = backbone

        # 1. 改进的FPN
        self.fpn = MarineEnhancedFPN_v3(
            features_channels=[512, 1024, 2048],
            out_channels=256,
            use_marine_enhance=True
        )

        # 2. 多尺度上下文增强模块
        if use_context_enhance:
            self.context_enhance = MultiScaleContextEnhancement(
                in_channels=256,
                out_channels=256,
                num_scales=3,
                use_dcn=True,
                use_channel_attn=True,
                use_spatial_attn=True
            )

        # 3. 海洋特征增强模块
        if use_marine_enhance:
            self.marine_enhance = MarineFeatureEnhancement(channels=256)

        # 4. 小目标专用检测头
        if use_small_target_head:
            self.small_target_head = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, embed_dim)
            )

            # 小目标查询嵌入
            self.small_query_embed = nn.Embedding(100, embed_dim)

        # 5. 自适应特征融合权重
        self.feature_fusion_weights = nn.Parameter(torch.ones(3))

        # 6. 其他模块保持不变
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

        # 7. 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 初始化特征融合权重
        nn.init.constant_(self.feature_fusion_weights, 1.0)

        # 初始化小目标查询
        if hasattr(self, 'small_query_embed'):
            nn.init.normal_(self.small_query_embed.weight, mean=0, std=0.01)

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # 1. 预处理
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        # 2. 骨干网络提取特征
        backbone_feats = self.backbone(images.tensors)

        # 3. FPN特征融合
        fpn_feats = self.fpn(backbone_feats)

        # 4. 多尺度上下文增强
        if self.use_context_enhance:
            enhanced_feats = self.context_enhance(fpn_feats)
        else:
            enhanced_feats = fpn_feats

        # 5. 海洋特征增强
        if self.use_marine_enhance:
            marine_enhanced_feats = []
            for feat in enhanced_feats:
                marine_feat = self.marine_enhance(feat)
                marine_enhanced_feats.append(marine_feat)
        else:
            marine_enhanced_feats = enhanced_feats

        # 6. 自适应特征融合
        if len(marine_enhanced_feats) > 1:
            weights = F.softmax(self.feature_fusion_weights[:len(marine_enhanced_feats)], dim=0)
            fused_feats = []
            for i, feat in enumerate(marine_enhanced_feats):
                weight = weights[i] if i < len(weights) else 1.0
                fused_feats.append(feat * weight)
        else:
            fused_feats = marine_enhanced_feats

        # 7. 小目标特征提取
        small_target_features = []
        if self.training and self.use_small_target_head:
            for feat in fused_feats:
                B, C, H, W = feat.shape
                if H * W > 32 * 32:  # 高分辨率特征图
                    small_feat = self.small_target_head(feat)
                    small_target_features.append(small_feat)

        # 8. 特征金字塔处理
        multi_level_feats = self.neck(fused_feats) if hasattr(self, 'neck') else fused_feats

        # 9. 创建mask和位置编码
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feature in multi_level_feats:
            multi_level_masks.append(F.interpolate(mask[None], size=feature.shape[-2:]).to(torch.bool)[0])
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # 10. 训练时的去噪处理
        if self.training:
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

        # 11. Transformer
        outputs_class, outputs_coord, enc_class, enc_coord, foreground_mask = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            noised_label_query,
            noised_box_query,
            attn_mask=attn_mask,
        )

        # 12. 小目标查询增强
        if self.training and self.use_small_target_head and len(small_target_features) > 0:
            # 将小目标特征注入到查询中
            small_query_feats = torch.stack(small_target_features, dim=0).mean(dim=0)
            small_query_feats = small_query_feats.unsqueeze(1)  # [B, 1, C]

            # 增强前10%的查询（假设这些更可能对应小目标）
            num_queries = outputs_class[-1].shape[1]
            num_small_queries = max(1, int(num_queries * 0.1))

            # 在查询特征上添加小目标偏置
            outputs_class[-1][:, :num_small_queries] += small_query_feats
            if self.aux_loss:
                for i in range(len(outputs_class) - 1):
                    outputs_class[i][:, :num_small_queries] += small_query_feats

            # 使用小目标查询嵌入
            small_query_embed = self.small_query_embed.weight.unsqueeze(0).repeat(outputs_class[-1].shape[0], 1, 1)
            outputs_class[-1][:, :num_small_queries] += 0.1 * small_query_embed[:, :num_small_queries]

        # 13. 去噪后处理
        if denoising_groups is not None and max_gt_num_per_image is not None:
            dn_metas = {
                "denoising_groups": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image,
            }
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas)

        # 14. 准备输出
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # 15. 计算损失
            loss_dict = self.criterion(output, targets)
            dn_losses = self.compute_dn_loss(dn_metas, targets)
            loss_dict.update(dn_losses)

            # 16. 使用增强的显著性损失
            feature_stride = [(
                images.tensors.shape[-2] / feature.shape[-2],
                images.tensors.shape[-1] / feature.shape[-1],
            ) for feature in multi_level_feats]

            # 使用EnhancedSalienceCriterion
            enhanced_focus_criterion = EnhancedSalienceCriterion(
                alpha=0.3,
                gamma=2.5,
                small_target_weight=2.0,
                marine_specific=True
            )

            focus_loss = enhanced_focus_criterion(
                foreground_mask, targets, feature_stride, images.image_sizes
            )
            loss_dict.update(focus_loss)

            # 17. 小目标专项损失
            if self.use_small_target_head:
                small_loss = self.compute_small_target_specific_loss(
                    output, targets, images.image_sizes
                )
                if small_loss:
                    loss_dict.update(small_loss)

            # 18. 损失重加权
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] = loss_dict[k] * weight_dict[k]

            return loss_dict

        # 19. 推理阶段
        detections = self.postprocessor(output, original_image_sizes)
        return detections

    def compute_small_target_specific_loss(
            self,
            output: Dict[str, Tensor],
            targets: List[Dict],
            image_sizes: List[Tuple[int, int]]
    ) -> Dict[str, Tensor]:
        """计算小目标专项损失"""
        if not targets:
            return {}

        # 获取预测结果
        pred_logits = output["pred_logits"]
        pred_boxes = output["pred_boxes"]

        losses = {}
        batch_size = len(targets)

        for i in range(batch_size):
            # 获取当前样本的目标
            target = targets[i]
            if len(target["boxes"]) == 0:
                continue

            # 计算目标面积
            boxes = target["boxes"]
            img_h, img_w = image_sizes[i]
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            boxes_pixel = box_ops._box_cxcywh_to_xyxy(boxes) * scale_factor.unsqueeze(0)
            areas = (boxes_pixel[:, 2] - boxes_pixel[:, 0]) * (boxes_pixel[:, 3] - boxes_pixel[:, 1])

            # 标记小目标
            small_indices = areas < 0.002 * (img_h * img_w)  # 面积小于0.2%

            if not small_indices.any():
                continue

            # 提取小目标的预测
            # 这里可以使用匈牙利匹配来匹配预测和目标
            # 简化版：直接计算小目标的分类损失
            small_labels = target["labels"][small_indices]

            # 为每个小目标找到最匹配的预测
            pred_scores = pred_logits[i].softmax(-1)  # [num_queries, num_classes]

            # 计算小目标匹配损失
            if len(small_labels) > 0:
                # 使用简单的最大分数匹配
                max_scores, pred_indices = pred_scores[:, small_labels].max(0)

                # 小目标分类损失
                small_target_loss = F.cross_entropy(
                    pred_logits[i][pred_indices],
                    small_labels,
                    reduction='mean'
                ) * 0.1  # 降低权重

                losses[f'loss_small_cls_{i}'] = small_target_loss

        # 聚合损失
        if losses:
            avg_loss = torch.stack(list(losses.values())).mean()
            return {'loss_small_target': avg_loss}

        return {}

    def _set_aux_loss(self, outputs_class, outputs_coord):
        """设置辅助损失"""
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        """去噪后处理"""
        if dn_metas is None:
            return outputs_class, outputs_coord

        # 提取去噪组信息
        denoising_groups = dn_metas["denoising_groups"]
        max_gt_num_per_image = dn_metas["max_gt_num_per_image"]

        # 分离去噪部分和匹配部分
        for i in range(len(outputs_class)):
            outputs_class[i] = outputs_class[i][:, denoising_groups:, :]
            outputs_coord[i] = outputs_coord[i][:, denoising_groups:, :]

        return outputs_class, outputs_coord

    def compute_dn_loss(self, dn_metas, targets):
        """计算去噪损失"""
        # 这里可以添加去噪相关的损失计算
        # 根据实际需要实现
        return {}


# 简化的SalienceDETR（原版）
class SalienceDETR(DNDETRDetector):
    """原始的SalienceDETR实现"""

    def __init__(
            self,
            backbone: nn.Module,
            neck: nn.Module,
            position_embedding: nn.Module,
            transformer: nn.Module,
            criterion: nn.Module,
            postprocessor: nn.Module,
            focus_criterion: nn.Module,
            num_classes: int = 91,
            num_queries: int = 1600,
            denoising_nums: int = 100,
            aux_loss: bool = True,
            min_size: int = None,
            max_size: int = None,
    ):
        super().__init__(min_size, max_size)
        # 基础参数
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        embed_dim = transformer.embed_dim

        # 基础结构
        self.backbone = backbone
        self.fpn = MarineEnhancedFPN_v3(
            features_channels=[512, 1024, 2048],
            out_channels=256,
            use_marine_enhance=False
        )
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

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # 原版实现
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        # 提取特征
        multi_level_feats = self.backbone(images.tensors)
        multi_level_feats = self.fpn(multi_level_feats)
        multi_level_feats = self.neck(multi_level_feats)

        multi_level_masks = []
        multi_level_position_embeddings = []
        for feature in multi_level_feats:
            multi_level_masks.append(F.interpolate(mask[None], size=feature.shape[-2:]).to(torch.bool)[0])
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        if self.training:
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

        # Transformer
        outputs_class, outputs_coord, enc_class, enc_coord, foreground_mask = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            noised_label_query,
            noised_box_query,
            attn_mask=attn_mask,
        )

        # 去噪后处理
        if denoising_groups is not None and max_gt_num_per_image is not None:
            dn_metas = {
                "denoising_groups": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image,
            }
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas)

        # 准备输出
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # 计算损失
            loss_dict = self.criterion(output, targets)
            dn_losses = self.compute_dn_loss(dn_metas, targets)
            loss_dict.update(dn_losses)

            feature_stride = [(
                images.tensors.shape[-2] / feature.shape[-2],
                images.tensors.shape[-1] / feature.shape[-1],
            ) for feature in multi_level_feats]

            focus_loss = self.focus_criterion(foreground_mask, targets, feature_stride, images.image_sizes)
            loss_dict.update(focus_loss)

            # 损失重加权
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] = loss_dict[k] * weight_dict[k]

            return loss_dict

        detections = self.postprocessor(output, original_image_sizes)
        return detections