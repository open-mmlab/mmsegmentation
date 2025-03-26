_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/hots_v1_640x480.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (640, 480)
data_preprocessor = dict(size=crop_size)
num_classes = 46
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    decode_head=dict(
        in_channels=320,
        num_classes=num_classes
    ),
    auxiliary_head=dict(
        in_channels=96,
        num_classes=num_classes
    )
)
