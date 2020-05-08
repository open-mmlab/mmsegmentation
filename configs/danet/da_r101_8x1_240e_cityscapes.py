_base_ = './da_r50_8x1_240e_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101_encoding-5be5422a.pth',
    backbone=dict(depth=101))
