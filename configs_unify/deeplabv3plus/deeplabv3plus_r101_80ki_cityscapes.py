_base_ = './deeplabv3plus_r50_80ki_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c-5fe8ded3.pth',
    backbone=dict(depth=101))
