_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', 'twins_schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/alt_gvt_large.pth',
    backbone=dict(
        type='Twins_alt_gvt',
        patch_size=4,
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_cfg=dict(type='LN'),
        depths=[2, 2, 18, 2],
        wss=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        extra_norm=True,
        drop_path_rate=0.3,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150),
    )

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
