_base_ = './van-b2_upernet_4xb2-160k_ade20k-512x512.py'
ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b5-in22k_3rdparty_20230522-b26134d7.pth'  # noqa
model = dict(
    backbone=dict(
        embed_dims=[96, 192, 480, 768],
        depths=[3, 3, 24, 3],
        init_cfg=dict(type='Pretrained', checkpoint=ckpt_path),
        drop_path_rate=0.4),
    decode_head=dict(in_channels=[96, 192, 480, 768], num_classes=150),
    auxiliary_head=dict(in_channels=480, num_classes=150))
