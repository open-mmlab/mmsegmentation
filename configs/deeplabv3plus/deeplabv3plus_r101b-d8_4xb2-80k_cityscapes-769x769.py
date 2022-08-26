_base_ = './deeplabv3plus_r50-d8_4xb2-80k_cityscapes-769x769.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101))
