_base_ = './van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py'

model = dict(
    backbone=dict(
        embed_dims=[96, 192, 384, 768],
        depths=[6, 6, 90, 6],
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/van_b6_22k.pth'),
        drop_path_rate=0.5),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150))
