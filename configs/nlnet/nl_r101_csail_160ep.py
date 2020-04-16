_base_ = './nl_r50_csail_160ep.py'
model = dict(
    pretrained='pretrain_model/resnet101c128_csail-159f67a3.pth',
    backbone=dict(depth=101))
