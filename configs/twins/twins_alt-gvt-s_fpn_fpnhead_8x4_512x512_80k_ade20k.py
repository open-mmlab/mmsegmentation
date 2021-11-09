_base_ = [
    '../_base_/models/twins_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/alt_gvt_small.pth',
    backbone=dict(
        type='ALTGVT',
        patch_size=4,
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_cfg=dict(type='LN'),
        depths=[2, 2, 10, 4],
        wss=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        extra_norm=True,
        drop_path_rate=0.2,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150),
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
