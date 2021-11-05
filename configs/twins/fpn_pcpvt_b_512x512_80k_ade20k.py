_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', 'twins_schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/pcpvt_base.pth',
    backbone=dict(
        type='Twins_pcpvt',
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_cfg=dict(type='LN'),
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.2,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150),
)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

data = dict(samples_per_gpu=4)
