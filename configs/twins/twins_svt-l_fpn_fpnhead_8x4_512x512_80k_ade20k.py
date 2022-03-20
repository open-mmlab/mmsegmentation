_base_ = ['./twins_svt-s_fpn_fpnhead_8x4_512x512_80k_ade20k.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/alt_gvt_large_20220308-fb5936f3.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3),
    neck=dict(in_channels=[128, 256, 512, 1024]),
)
