_base_ = [
    '../_base_/models/upernet_convnext.py',
    '../_base_/datasets/GFflood.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=2  # UAVflood二分类：background, flood
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=2
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(170, 170))
)

# 使用AMP优化器包装器以提高训练效率
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05
    ),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic'
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=20000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# 数据加载器配置，适配UAVflood数据集
train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=8, num_workers=8)
test_dataloader = val_dataloader

# 可选：设置随机种子以提高可复现性
randomness = dict(
    seed=42,
    deterministic=False,
)