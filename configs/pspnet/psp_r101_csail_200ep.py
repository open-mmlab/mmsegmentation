_base_ = './psp_r50_tv_200ep.py'
model = dict(
    pretrained='pretrain_model/resnet50c128_csail-0a46e9a7.pth',
    backbone=dict(depth=101, deep_stem=True, base_channels=128))
