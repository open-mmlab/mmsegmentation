_base_ = './psp_r50_tv_200ep.py'
norm_cfg = dict(type='PapeSyncBN', requires_grad=True)
model = dict(
    pretrained='pretrain_model/resnet50c128_hszhao-b3e6e229.pth',
    backbone=dict(deep_stem=True, base_channels=128, norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg))
