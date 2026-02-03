_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/SARflood.py',
    '../_base_/default_runtime.py'
]

crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        img_size=(256, 256),
        drop_path_rate=0.1,
        final_norm=True),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=5),  # warmup前5个epoch
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
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

randomness = dict(
    seed=42,
    deterministic=False,  # 如需完全可复现，设为True
)

# 设置日志按epoch显示
log_processor = dict(by_epoch=True)

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=8, num_workers=8)
test_dataloader = val_dataloader