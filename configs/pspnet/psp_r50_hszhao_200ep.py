_base_ = './psp_r50_tv_200ep.py'
model = dict(
    pretrained='pretrain_model/resnet50c128_hszhao-b3e6e229.pth',
    backbone=dict(deep_stem=True, base_channels=128))
