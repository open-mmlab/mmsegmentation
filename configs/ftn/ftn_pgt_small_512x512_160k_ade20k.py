_base_ = [
    '../_base_/models/ftn_pgt.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        group_nums=[64, 16, 1, 1],
        drop_path_rate=0.1),
    neck=dict(
        in_channels=[192, 384, 768],
        out_channels=512),
    decode_head=dict(
        in_channels=[512, 512, 512],
        channels=512,
        num_layers=[[1, 1, 1], [1, 1], [1]],   # [[0, 0, 0], [0, 0], [0]],   
        num_heads=4,
        sra_ratios=[[2, 2, 2], [2, 2], [2]], 
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=[0., 0., 0.],
        use_ape=False,
        num_classes=150),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,  # 256 for encoder_s3, 512 for neck_s3
        in_index=4,  # 1: encoder_s3,  4: neck_s3
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00004,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'cpe': dict(decay_mult=0.),
            'ape': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'decode_head': dict(lr_mult=2.),
            'auxiliary_head': dict(lr_mult=2.),
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

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)

