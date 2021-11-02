_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model=dict(
    backbone=dict(
        stdc_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='./pretrained/pretrained_stdc-813.pth'),
        )
    )
)