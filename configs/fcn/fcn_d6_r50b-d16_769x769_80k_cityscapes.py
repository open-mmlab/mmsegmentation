# Copyright (c) OpenMMLab. All rights reserved.
_base_ = './fcn_d6_r50-d16_769x769_80k_cityscapes.py'
model = dict(pretrained='torchvision://resnet50', backbone=dict(type='ResNet'))
