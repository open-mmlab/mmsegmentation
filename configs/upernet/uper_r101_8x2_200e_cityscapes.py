_base_ = './uper_r50_8x2_200e_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101c128_csail-159f67a3.pth',
    backbone=dict(depth=101))
