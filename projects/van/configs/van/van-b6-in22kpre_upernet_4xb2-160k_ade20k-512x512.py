_base_ = './van-b2_upernet_4xb2-160k_ade20k-512x512.py'
ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b6-in22k_3rdparty_20230522-5e5172a3.pth'  # noqa
model = dict(
    backbone=dict(
        embed_dims=[96, 192, 384, 768],
        depths=[6, 6, 90, 6],
        init_cfg=dict(type='Pretrained', checkpoint=ckpt_path),
        drop_path_rate=0.5),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150))
