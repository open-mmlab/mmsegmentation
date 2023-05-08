_base_ = './van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py'

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b0.pth')),
    decode_head=dict(in_channels=[32, 64, 160, 256], num_classes=150),
    auxiliary_head=dict(in_channels=160, num_classes=150))
