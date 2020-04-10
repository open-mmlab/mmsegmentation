_base_ = './psa_r50_hszhao_200ep.py'
model = dict(
    pretrained='pretrain_model/resnet101c128_hszhao-f9120436.pth',
    backbone=dict(depth=101))
