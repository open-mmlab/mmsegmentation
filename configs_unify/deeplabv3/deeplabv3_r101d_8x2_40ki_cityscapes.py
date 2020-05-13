_base_ = './deeplabv3_r50_8x2_40ki_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1d-3060aff9.pth',
    backbone=dict(type='ResNetV1d', depth=101, stem_channels=64))
