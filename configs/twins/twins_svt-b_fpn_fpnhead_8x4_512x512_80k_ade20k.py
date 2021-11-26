_base_ = ['./twins_svt-s_fpn_fpnhead_8x4_512x512_80k_ade20k.py']

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/alt_gvt_base.pth'),
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        depths=[2, 2, 18, 2]),
    neck=dict(in_channels=[96, 192, 384, 768]),
)
