_base_ = './psa_r50_8x2_200e_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101c128_hszhao-f9120436.pth',
    backbone=dict(depth=101))
