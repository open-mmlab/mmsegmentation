_base_ = './pspnet_r50-d8_4xb2-80k_cityscapes-769x769.py'
model = dict(pretrained='torchvision://resnet50', backbone=dict(type='ResNet'))
