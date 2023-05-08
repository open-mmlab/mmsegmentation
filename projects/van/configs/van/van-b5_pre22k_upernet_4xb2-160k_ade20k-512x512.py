_base_ = './van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py'

model = dict(
    backbone=dict(
        embed_dims=[96, 192, 480, 768],
        depths=[3, 3, 24, 3],
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/van_b5_22k.pth'),
        drop_path_rate=0.4),
    decode_head=dict(in_channels=[96, 192, 480, 768], num_classes=150),
    auxiliary_head=dict(in_channels=480, num_classes=150))
