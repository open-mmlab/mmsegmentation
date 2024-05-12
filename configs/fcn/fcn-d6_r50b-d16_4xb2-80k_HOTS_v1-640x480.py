_base_ = './fcn-d6_r50-d16_4xb2-80k_HOTS_v1-640x480.py'
model = dict(pretrained='torchvision://resnet50', backbone=dict(type='ResNet'))
