# Copyright (c) OpenMMLab. All rights reserved.
"""CM-DGSeg B0 configuration for Cityscapes.

Run command:
    python tools/train.py configs/cm-dgseg/cm_dgseg_b0_cityscapes.py
    python tools/test.py configs/cm-dgseg/cm_dgseg_b0_cityscapes.py <checkpoint> --eval mIoU
"""

_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
]

crop_size = (512, 1024)
num_classes = 19
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 0.0],
    std=[58.395, 57.12, 57.375, 1.0],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='DualStreamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone_rgb=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'),
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    backbone_disp=dict(
        type='MixVisionTransformer',
        in_channels=1,
        embed_dims=16,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 1, 3, 4],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.05),
    decode_head=dict(
        type='CMFSegFormerHead',
        in_channels=[32, 64, 160, 256],
        disp_in_channels=[16, 16, 48, 64],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        with_boundary=False,
        ignore_index=255,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                avg_non_ignore=True
            ),
            dict(
                type='LovaszLoss',
                per_image=False,
                reduction='none',
                loss_weight=0.4
            )
        ]
        ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(768, 768)))

# Data pipelines: disable photometric distortion to keep disparity channel clean.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCityscapesDisparity'),
    dict(type='ConcatRGBDispTo4Ch', keep_rgb_dtype=False),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCityscapesDisparity'),
    dict(type='ConcatRGBDispTo4Ch', keep_rgb_dtype=False, delete_disp=True),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCityscapesDisparity'),
    dict(type='ConcatRGBDispTo4Ch', keep_rgb_dtype=False, delete_disp=True),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in [0.5, 0.75, 1.0, 1.25, 1.5]
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ])
]

data_root = '/home/featurize/data/cityscapes'
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='CityscapesDataset',
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore', 'mAcc', 'mPrecision', 'mRecall'],
)

val_evaluator = test_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.),
            norm=dict(decay_mult=0.),
            head=dict(lr_mult=10.),
            backbone_disp=dict(lr_mult=1.0))))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000,save_best='mIoU', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

auto_scale_lr = dict(enable=False, base_batch_size=16)


load_from = './work_dirs/ZCDataset-Segformer-20251129/iter_80000.pth'

work_dir = './work_dirs/ZCDataset-Segformer-20251130-ft'