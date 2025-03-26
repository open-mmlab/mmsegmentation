_base_ = './fcn-d6_r50-d16_4xb2-40k_HOTS_v1-640x480.py'
load_from = "checkpoints/fcn_d6_r101b-d16_512x1024_80k_cityscapes_20210311_144305-3f2eb5b4.pth"
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101))
