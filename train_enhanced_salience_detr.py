import torch
import torch.nn as nn
from models.detectors.enhanced_salience_detr import EnhancedSalienceDETR
from models.bricks.enhanced_salience_criterion import EnhancedSalienceCriterion


# 初始化组件
def build_model(num_classes=91):
    # 1. 骨干网络
    from models.backbone import ResNet50Backbone
    backbone = ResNet50Backbone(pretrained=True)

    # 2. FPN
    from models.neck import FPN
    fpn = FPN(in_channels=[512, 1024, 2048], out_channels=256)

    # 3. Transformer
    from models.transformer import Transformer
    transformer = Transformer(
        embed_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
    )

    # 4. 位置编码
    from models.position_embedding import PositionEmbeddingSine
    position_embedding = PositionEmbeddingSine(num_pos_feats=128, normalize=True)

    # 5. 损失函数
    from models.criterion import SetCriterion
    criterion = SetCriterion(num_classes=num_classes)

    # 6. 后处理
    from models.postprocessor import PostProcess
    postprocessor = PostProcess()

    # 7. 显著性损失
    focus_criterion = EnhancedSalienceCriterion(
        alpha=0.3,
        gamma=2.5,
        small_target_weight=2.0,
        marine_specific=True,
    )

    # 8. 构建模型
    model = EnhancedSalienceDETR(
        backbone=backbone,
        neck=fpn,
        position_embedding=position_embedding,
        transformer=transformer,
        criterion=criterion,
        postprocessor=postprocessor,
        focus_criterion=focus_criterion,
        num_classes=num_classes,
        num_queries=1600,
        denoising_nums=100,
        aux_loss=True,
        use_context_enhance=True,
        use_small_target_head=True,
        use_marine_enhance=True,
    )

    return model


# 训练配置
def configure_training(model, lr=1e-4):
    # 参数分组
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': lr * 0.1},
        {'params': model.fpn.parameters(), 'lr': lr},
        {'params': model.context_enhance.parameters(), 'lr': lr},
        {'params': model.transformer.parameters(), 'lr': lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    return optimizer, scheduler


# 训练循环
def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播
        loss_dict = model(images, targets)

        # 计算总损失
        total_loss_current = sum(loss for loss in loss_dict.values())

        # 反向传播
        optimizer.zero_grad()
        total_loss_current.backward()
        optimizer.step()

        total_loss += total_loss_current.item()

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}: Loss = {total_loss_current.item():.4f}')

    return total_loss / len(data_loader)


# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = build_model(num_classes=91)
    model = model.to(device)

    # 配置优化器
    optimizer, scheduler = configure_training(model, lr=1e-4)

    # 数据加载器
    from data.datasets import build_dataset
    train_dataset = build_dataset('coco', split='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4
    )

    # 训练
    num_epochs = 100
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, None, device)
        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}: Average Loss = {avg_loss:.4f}')

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    main()