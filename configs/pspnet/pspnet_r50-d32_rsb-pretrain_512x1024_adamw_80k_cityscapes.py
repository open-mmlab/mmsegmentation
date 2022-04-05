_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa
model = dict(
    pretrained=None,
    backbone=dict(
        type='ResNet',
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 2, 2)))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0005, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[60000, 72000],
    by_epoch=False)
