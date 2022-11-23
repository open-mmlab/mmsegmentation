_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/imagenets.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

model = dict(
    pretrained='./pretrain/sere_pretrained_vit_small_ep100_mmcls.pth',
    backbone=dict(
        _delete_=True,
        type='VisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        in_channels=3,
        embed_dims=384,
        num_layers=12,
        num_heads=6,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        final_norm=True,
        interpolate_mode='bicubic'),
    decode_head=dict(
        in_channels=384,
        channels=384,
        num_convs=0,
        dropout_ratio=0.0,
        num_classes=920,
        ignore_index=1000,
        downsample_label_ratio=8,
        init_cfg=dict(
            type='TruncNormal', std=2e-5, override=dict(name='conv_seg'))),
    auxiliary_head=None)

optimizer = dict(
    _delete_=True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        num_layers=12, decay_rate=0.50, decay_type='layer_wise'))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=180,
    warmup_ratio=1e-6,
    min_lr=1e-6,
    by_epoch=False)

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 32 images per GPU
data = dict(samples_per_gpu=32)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=3600)
checkpoint_config = dict(by_epoch=False, interval=3600)
evaluation = dict(interval=360, metric='mIoU', pre_eval=True)
