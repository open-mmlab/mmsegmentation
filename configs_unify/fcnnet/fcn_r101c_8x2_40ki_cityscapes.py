_base_ = './fcn_r50_8x2_40ki_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c-5fe8ded3.pth',
    backbone=dict(type='ResNetV1c', depth=101, stem_channels=64))
