_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/UAVflood.py',
    '../_base_/default_runtime.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)

# 修改为epoch训练，训练100个epoch
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=5),  # warmup前5个epoch
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=5,
        end=100,
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
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=10,  # 每10个epoch保存一次
        max_keep_ckpts=3,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(
    seed=42,
    deterministic=False,  # 如需完全可复现，设为True
)
