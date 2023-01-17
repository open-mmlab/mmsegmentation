_base_ = [
    '../../../configs/deeplabv3plus/deeplabv3plus_\
        r50-d8_4xb2-80k_cityscapes-512x1024.py',
    './_base_/datasets/mapillary_v2_0.py',  # v2.0 labels
]

custom_imports = dict(imports=[
    'projects.Mapillary_dataset.mmseg.datasets.mapillary_v1_2',
    'projects.Mapillary_dataset.mmseg.datasets.mapillary_v2_0',
])
model = dict(
    backbone=dict(type='ResNet', depth=101),
    decode_head=dict(
        # num_classes=66,  # v1.2
        num_classes=124,  # v2.0
    ),
    auxiliary_head=dict(
        # num_classes=66,  # v1.2
        num_classes=124,  # v2.0
        train_cfg=dict(
            type='IterBasedTrainLoop', max_iters=240000, val_interval=24000)))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=None)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=240000,
        by_epoch=False)
]
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=24000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
