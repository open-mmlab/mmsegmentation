_base_ = [
    '../_base_/models/van_fpn.py',
    '../../../../configs/_base_/datasets/ade20k.py',
    '../../../../configs/_base_/default_runtime.py',
]
custom_imports = dict(imports=['projects.van.backbones'])


# checkpoint_file = 'https://cloud.tsinghua.edu.cn/d/0100f0cea37d41ba8d08/files/?p=%2Fvan_b2.pth&dl=1'
checkpoint_file = 'pretrained/van_b2.pth'
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 12, 3],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        drop_path_rate=0.2),
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(num_classes=150))

train_dataloader = dict(
    batch_size=4
)


# we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
gpu_multiples = 2
max_iters = 80000 // gpu_multiples
interval = 8000 // gpu_multiples
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW',
                   lr=0.0001 * gpu_multiples,
                   betas=(0.9, 0.999),
                   weight_decay=0.0001),
    clip_grad=None
)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        power=0.9,
        eta_min=0.0,
        begin=0,
        end=max_iters,
        by_epoch=False,
    )
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=interval),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
