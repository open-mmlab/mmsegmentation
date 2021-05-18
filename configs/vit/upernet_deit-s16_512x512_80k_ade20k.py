_base_ = [
    '../_base_/models/upernet_vit-b16.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',  # noqa
    backbone=dict(num_heads=6, embed_dim=384, drop_path_rate=0.1),
    neck=dict(in_channels=[384], out_channels=384),
    decode_head=dict(num_classes=150, in_channels=[384, 384, 384, 384]),
    auxiliary_head=dict(num_classes=150, in_channels=384))  # yapf: disable

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
