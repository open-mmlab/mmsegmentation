# ------------------------------------------------------------
# SegFormer B0 baseline for Cityscapes (for comparison with CM-DGSeg B0)
# Modified from your original B2 config
# ------------------------------------------------------------

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

crop_size = (512, 1024)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

data_root = '/home/featurize/data/cityscapes'
dataset_type = 'CityscapesDataset'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=3,
        save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'),
)

default_scope = 'mmseg'

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

log_level = 'INFO'
log_processor = dict(by_epoch=False)

# ------------------------------------------------------------
# SegFormer-B0 configuration
# ------------------------------------------------------------
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='MixVisionTransformer',
        # -------- change to B0 --------
        init_cfg=dict(type='Pretrained',
                      checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'),
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        mlp_ratio=4,
        qkv_bias=True,

        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
    ),

    decode_head=dict(
        type='SegformerHead',
        # -------- B0 decode head channels --------
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,

        dropout_ratio=0.1,
        num_classes=19,

        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),

    pretrained=None,

    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        crop_size=crop_size,
        stride=(768, 768),
    )
)

norm_cfg = dict(type='BN', requires_grad=True)

# ------------------------------------------------------------
# Optimizer (same as B2 version)
# ------------------------------------------------------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0),
        )
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False),
]

randomness = dict(seed=0)

resume = False

# ------------------------------------------------------------
# Train / Val / Test
# ------------------------------------------------------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ------------------- dataloaders -----------------------------

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),

    dataset=dict(
        type='CityscapesDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'),
        reduce_zero_label=True,

        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            # dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),

    dataset=dict(
        type='CityscapesDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'),
        reduce_zero_label=True,

        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='PackSegInputs'),
        ]
    )
)

test_dataloader = val_dataloader

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore', 'mAcc', 'mPrecision', 'mRecall'],
)

val_evaluator = test_evaluator

# ------------------------------------------------------------
# TTA
# ------------------------------------------------------------
tta_model = dict(type='SegTTAModel')

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=s, keep_ratio=True)
                for s in [0.5, 0.75, 1.0, 1.25, 1.5]
            ],
            [
                dict(type='RandomFlip', direction='horizontal', prob=0.0),
                dict(type='RandomFlip', direction='horizontal', prob=1.0),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ]
        ]
    )
]

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    type='SegLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends,
)

work_dir = './work_dirs/ZCDataset-SegformerB0-compare-20251126'

