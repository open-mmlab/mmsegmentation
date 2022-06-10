_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 1024)
preprocess_cfg = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa
model = dict(
    preprocess_cfg=preprocess_cfg,
    pretrained=None,
    backbone=dict(
        type='ResNet',
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0005, weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
default_hooks = dict(
    optimizer=dict(
        _delete_=True,
        type='OptimizerHook',
        grad_clip=dict(max_norm=1, norm_type=2)))
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=1000,
        end=80000,
        by_epoch=False,
        milestones=[60000, 72000],
    )
]
