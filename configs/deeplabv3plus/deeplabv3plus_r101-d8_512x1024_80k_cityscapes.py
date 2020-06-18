_base_ = './deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c_trick-e67eebb6.pth',
    backbone=dict(depth=101))
