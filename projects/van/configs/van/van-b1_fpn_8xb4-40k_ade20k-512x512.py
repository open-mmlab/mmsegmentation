_base_ = './van-b2_fpn_8xb4-40k_ade20k-512x512.py'

# model settings
model = dict(
    backbone=dict(
        depths=[2, 2, 4, 2],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b1.pth')))
