_base_ = [
    '../_base_/models/icnet_r50-d8.py',
    '../_base_/datasets/cityscapes_832x832.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'))))
