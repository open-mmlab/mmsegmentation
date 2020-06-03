_base_ = './psa_r50_8x2_50e_voc12aug.py'
model = dict(
    pretrained='pretrain_model/resnet101c128_hszhao-f9120436.pth',
    backbone=dict(depth=101))
