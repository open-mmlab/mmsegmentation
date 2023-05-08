_base_ = './van-b2_fpn_8xb4-40k_ade20k-512x512.py'

model = dict(
    backbone=dict(
        embed_dims=[32, 64, 160, 256],
        depths=[3, 3, 5, 2],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b0.pth')),
    neck=dict(in_channels=[32, 64, 160, 256]))
