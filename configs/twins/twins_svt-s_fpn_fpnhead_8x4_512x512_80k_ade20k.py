_base_ = [
    '../_base_/models/twins_pcpvt-s_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
backbone_norm_cfg = dict(type='LN')
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SVT',
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/alt_gvt_small.pth'),
        patch_size=4,
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        depths=[2, 2, 10, 4],
        windiow_size=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        norm_after_stage=True,
        drop_path_rate=0.2),
    neck=dict(in_channels=[64, 128, 256, 512], out_channels=256, num_outs=4),
    decode_head=dict(num_classes=150),
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
