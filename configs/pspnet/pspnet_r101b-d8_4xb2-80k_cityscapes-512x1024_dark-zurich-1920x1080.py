_base_ = './pspnet_r50-d8_4xb2-80k_cityscapes-512x1024_dark-zurich-1920x1080.py'  # noqa
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101))
