_base_ = './van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py'

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        depths=[3, 5, 27, 3],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b3.pth'),
        drop_path_rate=0.3))
