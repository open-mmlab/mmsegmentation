_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/mapillary_v1_65.py',
    '../_base_/default_runtime.py',
]

crop_size = (1280, 1280)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(depth=50),
    decode_head=dict(num_classes=65),
    auxiliary_head=dict(num_classes=65))

iters = 300000
# optimizer
optimizer = dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]

# training schedule for 300k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

train_dataloader = dict(batch_size=2)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=8)
