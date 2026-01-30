# 配置文件
model_cfg = {
    'type': 'EnhancedSalienceDETR',

    # 骨干网络
    'backbone': {
        'type': 'ResNet50',
        'pretrained': True,
        'frozen_stages': 1,
    },

    # FPN
    'fpn': {
        'type': 'MarineEnhancedFPN_v3',
        'features_channels': [512, 1024, 2048],
        'out_channels': 256,
        'use_marine_enhance': True,
    },

    # 多尺度上下文增强
    'context_enhance': {
        'enabled': True,
        'in_channels': 256,
        'out_channels': 256,
        'num_scales': 3,
        'use_dcn': True,
        'use_channel_attn': True,
        'use_spatial_attn': True,
    },

    # 海洋特征增强
    'marine_enhance': {
        'enabled': True,
        'channels': 256,
    },

    # 小目标专用头
    'small_target_head': {
        'enabled': True,
    },

    # Transformer
    'transformer': {
        'embed_dim': 256,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
    },

    # 损失函数
    'criterion': {
        'type': 'SetCriterion',
        'num_classes': 91,
        'weight_dict': {
            'loss_ce': 1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
        },
    },

    'focus_criterion': {
        'type': 'EnhancedSalienceCriterion',
        'alpha': 0.3,
        'gamma': 2.5,
        'small_target_weight': 2.0,
        'marine_specific': True,
    },

    # 训练参数
    'num_queries': 1600,
    'denoising_nums': 100,
    'aux_loss': True,
}