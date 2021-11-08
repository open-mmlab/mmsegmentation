_base_ = [
    'twins_alt-gvt-s_fpn_fpnhead_8x4_512x512_80k_ade20k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/alt_gvt_base.pth',
    backbone=dict(
        type='Twins_alt_gvt',
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        depths=[2, 2, 18, 2]),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768]),
    decode_head=dict(num_classes=150),
)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
