norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(256, 256),
        patch_size=16,
        in_channels=4,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-06),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        out_shape='NCHW',
        interpolate_mode='bicubic',
        final_norm=True),
    decode_head=dict(
        type='LinearTransformerHead',
        in_channels=768,
        in_index=3,
        channels=768,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(171, 171)))
dataset_type = 'VaihingenDataset'
data_root = 'G:/Datasets/Vaihingen/ISPRS_semantic_labeling_Vaihingen'
img_norm_cfg = dict(
    mean=[120.476, 81.7993, 81.1927, 30.672],
    std=[54.8465, 39.3214, 37.9183, 38.0866],
    to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 1536), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[120.476, 81.7993, 81.1927, 30.672],
        std=[54.8465, 39.3214, 37.9183, 38.0866],
        to_rgb=False),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1536),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[120.476, 81.7993, 81.1927, 30.672],
                std=[54.8465, 39.3214, 37.9183, 38.0866],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='VaihingenDataset',
        data_root='G:/Datasets/Vaihingen/ISPRS_semantic_labeling_Vaihingen',
        img_dir='with_nDSM/tra_val',
        ann_dir='annotations/tra_val',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='unchanged'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(
                type='Resize', img_scale=(2048, 1536), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[120.476, 81.7993, 81.1927, 30.672],
                std=[54.8465, 39.3214, 37.9183, 38.0866],
                to_rgb=False),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='VaihingenDataset',
        data_root='G:/Datasets/Vaihingen/ISPRS_semantic_labeling_Vaihingen',
        img_dir='with_nDSM/testing',
        ann_dir='annotations/testing',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='unchanged'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1536),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[120.476, 81.7993, 81.1927, 30.672],
                        std=[54.8465, 39.3214, 37.9183, 38.0866],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='VaihingenDataset',
        data_root='G:/Datasets/Vaihingen/ISPRS_semantic_labeling_Vaihingen',
        img_dir='with_nDSM/testing',
        ann_dir='annotations/testing',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='unchanged'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1536),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[120.476, 81.7993, 81.1927, 30.672],
                        std=[54.8465, 39.3214, 37.9183, 38.0866],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_embed=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')
work_dir = './segmenter'
gpu_ids = range(0, 1)
