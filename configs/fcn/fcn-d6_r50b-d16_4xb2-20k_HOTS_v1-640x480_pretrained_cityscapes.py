_base_ = './fcn-d6_r50-d16_4xb2-20k_HOTS_v1-640x480.py'
load_from = "checkpoints/fcn_d6_r50b-d16_512x1024_80k_cityscapes_20210311_125550-6a0b62e9.pth"
model = dict(pretrained='torchvision://resnet50', backbone=dict(type='ResNet'))
