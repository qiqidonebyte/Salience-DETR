from torch.nn import functional as F
import torch


def diou_loss(boxes1, boxes2):
    """
    Distance-IoU Loss (DIoU) - 比GIoU收敛更快，精度更高
    参考: Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression (AAAI 2020)
    boxes format: (center_x, center_y, w, h) - normalized coordinates
    """
    # Convert to (x1, y1, x2, y2) for IoU calculation
    boxes1_xyxy = torch.stack([
        boxes1[..., 0] - boxes1[..., 2] / 2,
        boxes1[..., 1] - boxes1[..., 3] / 2,
        boxes1[..., 0] + boxes1[..., 2] / 2,
        boxes1[..., 1] + boxes1[..., 3] / 2,
    ], dim=-1)
    
    boxes2_xyxy = torch.stack([
        boxes2[..., 0] - boxes2[..., 2] / 2,
        boxes2[..., 1] - boxes2[..., 3] / 2,
        boxes2[..., 0] + boxes2[..., 2] / 2,
        boxes2[..., 1] + boxes2[..., 3] / 2,
    ], dim=-1)
    
    # Calculate IoU
    inter_x1 = torch.max(boxes1_xyxy[..., 0], boxes2_xyxy[..., 0])
    inter_y1 = torch.max(boxes1_xyxy[..., 1], boxes2_xyxy[..., 1])
    inter_x2 = torch.min(boxes1_xyxy[..., 2], boxes2_xyxy[..., 2])
    inter_y2 = torch.min(boxes1_xyxy[..., 3], boxes2_xyxy[..., 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-7)
    
    # Calculate center distance (normalized)
    center1 = boxes1[..., :2]
    center2 = boxes2[..., :2]
    center_distance = torch.sum((center1 - center2) ** 2, dim=-1)
    
    # Calculate diagonal distance of enclosing box (normalized)
    enclose_x1 = torch.min(boxes1_xyxy[..., 0], boxes2_xyxy[..., 0])
    enclose_y1 = torch.min(boxes1_xyxy[..., 1], boxes2_xyxy[..., 1])
    enclose_x2 = torch.max(boxes1_xyxy[..., 2], boxes2_xyxy[..., 2])
    enclose_y2 = torch.max(boxes1_xyxy[..., 3], boxes2_xyxy[..., 3])
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    # DIoU = IoU - (center_distance^2) / (enclose_diagonal^2)
    diou = iou - (center_distance / (enclose_diagonal + 1e-7))
    return 1 - diou


def ciou_loss(boxes1, boxes2):
    """
    Complete-IoU Loss (CIoU) - 考虑长宽比，比DIoU更精确
    参考: Complete-IoU Loss: Faster and Better Learning for Bounding Box Regression (AAAI 2020)
    boxes format: (center_x, center_y, w, h) - normalized coordinates
    """
    # Convert to (x1, y1, x2, y2) for IoU calculation
    boxes1_xyxy = torch.stack([
        boxes1[..., 0] - boxes1[..., 2] / 2,
        boxes1[..., 1] - boxes1[..., 3] / 2,
        boxes1[..., 0] + boxes1[..., 2] / 2,
        boxes1[..., 1] + boxes1[..., 3] / 2,
    ], dim=-1)
    
    boxes2_xyxy = torch.stack([
        boxes2[..., 0] - boxes2[..., 2] / 2,
        boxes2[..., 1] - boxes2[..., 3] / 2,
        boxes2[..., 0] + boxes2[..., 2] / 2,
        boxes2[..., 1] + boxes2[..., 3] / 2,
    ], dim=-1)
    
    # Calculate IoU
    inter_x1 = torch.max(boxes1_xyxy[..., 0], boxes2_xyxy[..., 0])
    inter_y1 = torch.max(boxes1_xyxy[..., 1], boxes2_xyxy[..., 1])
    inter_x2 = torch.min(boxes1_xyxy[..., 2], boxes2_xyxy[..., 2])
    inter_y2 = torch.min(boxes1_xyxy[..., 3], boxes2_xyxy[..., 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-7)
    
    # Calculate center distance (normalized)
    center1 = boxes1[..., :2]
    center2 = boxes2[..., :2]
    center_distance = torch.sum((center1 - center2) ** 2, dim=-1)
    
    # Calculate diagonal distance of enclosing box (normalized)
    enclose_x1 = torch.min(boxes1_xyxy[..., 0], boxes2_xyxy[..., 0])
    enclose_y1 = torch.min(boxes1_xyxy[..., 1], boxes2_xyxy[..., 1])
    enclose_x2 = torch.max(boxes1_xyxy[..., 2], boxes2_xyxy[..., 2])
    enclose_y2 = torch.max(boxes1_xyxy[..., 3], boxes2_xyxy[..., 3])
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    # Calculate aspect ratio consistency
    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-7)) - 
        torch.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-7)), 2
    )
    alpha = v / (1 - iou + v + 1e-7)
    
    # CIoU = IoU - (center_distance^2) / (enclose_diagonal^2) - alpha * v
    ciou = iou - (center_distance / (enclose_diagonal + 1e-7)) - alpha * v
    return 1 - ciou


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    # 将输入通过sigmoid函数转换为概率
    prob = inputs.sigmoid()
    # 确保target_score与inputs的dtype相同，以便后续计算
    target_score = targets.to(inputs.dtype)
    # 计算focal loss中的调制权重，用于减少易分类样本的权重，增加难分类样本的权重
    # alpha是一个用于平衡正负样本的超参数
    # gamma是一个调节模型关注难易样本的超参数
    weight = (1 - alpha) * prob ** gamma * (1 - targets) + targets * alpha * (1 - prob) ** gamma
    # according to original implementation, sigmoid_focal_loss keep gradient on weight
    # 计算二元交叉熵损失，这里使用reduction="none"来避免在损失求和时自动应用mean
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, reduction="none")
    # 将损失与调制权重相乘，得到最终的focal损失
    loss = loss * weight
    # we use sum/num to replace mean to avoid NaN
    # 通过sum/num的方式替换mean函数来计算平均损失，避免NaN问题
    # max函数确保在不同设备上都能正确处理损失的维度
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def vari_sigmoid_focal_loss(inputs, targets, gt_score, num_boxes, alpha: float = 0.25, gamma: float = 2):
    # 将输入通过sigmoid函数转换为概率，并且.detach()来阻止梯度传递
    prob = inputs.sigmoid().detach()  # pytorch version of RT-DETR has detach while paddle version not
    # gt_score是ground truth的分数，这里假设它是一个与targets相同shape的tensor
    # 将gt_score与targets相乘，并增加unsqueeze操作以匹配后续计算的维度
    target_score = targets * gt_score.unsqueeze(-1)
    # 计算focal loss的调制权重
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + target_score
    # 计算二元交叉熵损失，weight参数用于对损失进行加权
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    # 计算平均损失，处理方式与sigmoid_focal_loss相同
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def ia_bce_loss(inputs, targets, gt_score, num_boxes, k: float = 0.25, alpha: float = 0, gamma: float = 2):
    # 将输入通过sigmoid函数转换为概率，.detach()阻止梯度传递
    prob = inputs.sigmoid().detach()
    # 计算iou_aware_score，这是一个根据预测概率和gt_score计算的分数
    # calculate iou_aware_score and constrain the value following original implementation
    iou_aware_score = prob ** k * gt_score.unsqueeze(-1) ** (1 - k)
    # 将iou_aware_score限制在[0.01, 1]范围内，避免取值过小或过大
    iou_aware_score = iou_aware_score.clamp(min=0.01)
    # 将iou_aware_score与targets相乘，得到加权的目标分数
    target_score = targets * iou_aware_score
    # 计算focal loss的调制权重，alpha在这里的用途与之前略有不同
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + targets
    # 计算加权的二元交叉熵损失
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    # 计算平均损失
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes
