# 模型配置文件
MODEL_CONFIGS = {
    'salience_detr_resnet50': {
        'path': 'configs/salience_detr/salience_detr_resnet50_800_1333.py',
        'backbone': 'resnet50',
        'num_queries': 1600,
        'description': '原始SalienceDETR (ResNet50)'
    },
    'enhanced_salience_detr': {
        'path': 'configs/enhanced_salience_detr.py',
        'backbone': 'resnet50',
        'num_queries': 1600,
        'description': '增强版SalienceDETR (多尺度上下文增强)',
        'features': ['context_enhance', 'marine_enhance', 'small_target_head']
    },
    'salience_detr_swin_l': {
        'path': 'configs/salience_detr/salience_detr_swin_l_800_1333.py',
        'backbone': 'swin_l',
        'num_queries': 1600,
        'description': 'SalienceDETR (Swin-L)'
    }
}

# NPU优化配置
NPU_OPTIMIZATIONS = {
    'performance': {
        'mixed_precision': 'bf16',
        'use_apex': True,
        'opt_level': 'O2',
        'batch_size': 8,
        'num_workers': 8,
        'pin_memory': False
    },
    'memory': {
        'mixed_precision': 'bf16',
        'use_apex': True,
        'opt_level': 'O1',
        'batch_size': 4,
        'num_workers': 4,
        'pin_memory': False
    },
    'accuracy': {
        'mixed_precision': 'no',
        'use_apex': False,
        'opt_level': 'O0',
        'batch_size': 2,
        'num_workers': 8,
        'pin_memory': True
    }
}