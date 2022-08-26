_base_ = './pspnet_r50-d8_4xb4-80k_loveda-512x512.py'
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet18_v1c')),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
