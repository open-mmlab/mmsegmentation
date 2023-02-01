_base_ = './fcn-d6_r50b-d16_4xb2-80k_cityscapes-769x769.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101))
