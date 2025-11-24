from torch.nn import functional as F


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
