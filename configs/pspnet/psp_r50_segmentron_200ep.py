_base_ = './psp_r50_tv_200ep.py'
model = dict(
    pretrained='pretrain_model/resnet50c128_segmentron-8f647d7a.pth',
    backbone=dict(deep_stem=True, base_channels=128))
