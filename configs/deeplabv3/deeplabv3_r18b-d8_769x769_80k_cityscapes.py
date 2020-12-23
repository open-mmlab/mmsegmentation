_base_ = './deeplabv3_r50-d8_769x769_80k_cityscapes.py'
model = dict(
    type='ResNet',
    pretrained='torchvision://resnet18',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
