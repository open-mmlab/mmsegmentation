_base_ = ['./twins_svt-s_fpn_fpnhead_8x4_512x512_80k_ade20k.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/alt_gvt_base_20220308-1b7eb711.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        depths=[2, 2, 18, 2]),
    neck=dict(in_channels=[96, 192, 384, 768]),
)
