_base_ = ['./twins_svt-s_uperhead_8x2_512x512_160k_ade20k.py']
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/alt_gvt_large.pth'),
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3),
    decode_head=dict(in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(in_channels=512))
