
_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (224, 224)

model = dict(
    backbone=dict(
        type='EfficientNet',
        strides=(1, 2, 2, 2, 1, 2, 1),
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='Swish'),
        with_cp=False,
        pretrained=None,
        norm_eval=False,
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=150,
    ),
    auxiliary_head=dict(in_channels=512, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)
