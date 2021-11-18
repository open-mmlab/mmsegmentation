_base_ = ['twins_alt-gvt-s_uperhead_8x2_512x512_160k_ade20k.py']
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/alt_gvt_base.pth',
    backbone=dict(
        type='ALTGVT',
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        depths=[2, 2, 18, 2]),
    decode_head=dict(in_channels=[96, 192, 384, 768]),
    auxiliary_head=dict(in_channels=384))

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
