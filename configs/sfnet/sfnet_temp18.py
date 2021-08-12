_base_ = './sfnet_temp.py'
model = dict(
    pretrained=None,
    backbone=dict(type='ResNetV1c', depth=18, strides=(1, 2, 2, 2)),
    decode_head=dict(
        in_channels=512,
        channels=128,
        fpn_inplanes=[64, 128, 256, 512],
        fpn_dim=128,
    ),
)
