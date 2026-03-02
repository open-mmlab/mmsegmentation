_base_ = [
    '../_base_/models/upernet_convnext.py',
        '../_base_/datasets/SARflood.py',
    '../_base_/default_runtime.py'
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
        by_epoch=True,
        begin=0,
        end=5  # warmup前5个epoch
    ),
    dict(
        type='PolyLR',
        power=1.0,
        begin=5,
        end=100,
        eta_min=0.0,
        by_epoch=True,
    )
]

# 使用EpochBasedTrainLoop训练100个epoch
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10)  # 每10个epoch验证一次

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 修改hooks为基于epoch
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=10,  # 每10个epoch保存一次
        max_keep_ckpts=3,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# 设置日志按epoch显示
log_processor = dict(by_epoch=True)

# 数据加载器配置，适配UAVflood数据集
train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=8, num_workers=8)
test_dataloader = val_dataloader

# 可选：设置随机种子以提高可复现性
randomness = dict(
    seed=42,
    deterministic=False,
)