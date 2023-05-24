_base_ = './van-b2_fpn_8xb4-40k_ade20k-512x512.py'
ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b0_3rdparty_20230522-956f5e0d.pth'  # noqa
model = dict(
    backbone=dict(
        embed_dims=[32, 64, 160, 256],
        depths=[3, 3, 5, 2],
        init_cfg=dict(type='Pretrained', checkpoint=ckpt_path)),
    neck=dict(in_channels=[32, 64, 160, 256]))
