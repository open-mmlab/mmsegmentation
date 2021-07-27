_base_ = './sfnet_r50-d8_512x1024_80k_cityscapes.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        fpn_inplanes=[64, 128, 256, 512],
        fpn_dim=64,
    ),
)
