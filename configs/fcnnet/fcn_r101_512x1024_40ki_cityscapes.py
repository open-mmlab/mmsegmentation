_base_ = './fcn_r50_512x1024_40ki_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c_trick-e67eebb6.pth',
    backbone=dict(depth=101))
