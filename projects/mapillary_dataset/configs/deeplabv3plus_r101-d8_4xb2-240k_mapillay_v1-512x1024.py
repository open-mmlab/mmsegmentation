_base_ = [
    '../../../configs/_base_/models/deeplabv3plus_r50-d8.py',
    './_base_/datasets/mapillary_v1_2.py', '../../../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['projects.mapillary_dataset.mmseg.datasets.mapillary'])

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(num_classes=66),
    auxiliary_head=dict(num_classes=66))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=240000,
        by_epoch=False)
]
# training schedule for 40k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=240000, val_interval=24000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=24000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
