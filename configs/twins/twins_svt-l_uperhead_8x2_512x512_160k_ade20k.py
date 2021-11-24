_base_ = ['twins_svt-s_uperhead_8x2_512x512_160k_ade20k.py']
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SVT',
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/alt_gvt_large.pth'),
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        depths=[1, 1, 9, 1],
        drop_path_rate=0.3),
    decode_head=dict(in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(in_channels=512))

data = dict(samples_per_gpu=2, workers_per_gpu=2)
