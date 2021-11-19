_base_ = ['twins_altgvt-s_fpn_fpnhead_8x4_512x512_80k_ade20k.py']

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/alt_gvt_base.pth',
    backbone=dict(
        type='ALTGVT',
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        depths=[2, 2, 18, 2]),
    neck=dict(in_channels=[96, 192, 384, 768]),
    decode_head=dict(num_classes=150),
)
