_base_ = './da_r50_encoding_240ep.py'
model = dict(
    pretrained='pretrain_model/resnet101_encoding-5be5422a.pth',
    backbone=dict(depth=101))
