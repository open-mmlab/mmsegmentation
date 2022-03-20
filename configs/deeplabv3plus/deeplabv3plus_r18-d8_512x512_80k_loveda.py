_base_ = './deeplabv3plus_r50-d8_512x512_80k_loveda.py'
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet18_v1c')),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
