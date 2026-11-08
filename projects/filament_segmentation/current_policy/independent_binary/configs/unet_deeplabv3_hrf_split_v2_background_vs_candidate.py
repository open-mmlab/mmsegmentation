"""
Train/validation only config for:
    background vs candidate_filament

Dataset:
    /mnt/sp4t/fujiwara.r/split_v2

IMPORTANT:
- train and val only
- test is NOT used
- existing 0/1/2/255 masks are remapped in memory
- DeepLabV3 + UNet base model is kept unchanged
- HRF pretrained checkpoint is kept unchanged
"""

_base_ = [
    '../../../../../configs/_base_/default_runtime.py',
]

default_scope = 'mmseg'

custom_imports = dict(
    imports=[
        'mmseg.models',
        'projects.filament_segmentation.current_policy.independent_binary.datasets.binary_filament_transforms_split_v2',
    ],
    allow_failed_imports=False,
)

# =============================================================================
# Dataset
# =============================================================================

dataset_type = 'BaseSegDataset'
data_root = '/mnt/sp4t/fujiwara.r/split_v2'
crop_size = (256, 256)
reduce_zero_label = False

metainfo = dict(
    classes=(
        'background',
        'candidate_filament',
    ),
    palette=[
        [0, 0, 0],
        [255, 255, 255],
    ],
)

# =============================================================================
# HRF pretrained checkpoint
# =============================================================================

load_from = (
    'checkpoints/'
    'deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_'
    '256x256_40k_hrf_20211210_202032-59daf7a4.pth'
)

resume = False

# =============================================================================
# Preprocessing
# =============================================================================

data_preprocessor = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
)

norm_cfg = dict(
    type='BN',
    requires_grad=True,
)

# =============================================================================
# Model
# DeepLabV3 + UNet backbone: unchanged two-class architecture
# =============================================================================

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        downsamples=(True, True, True, True),
        conv_cfg=None,
        norm_cfg=norm_cfg,
        norm_eval=False,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        with_cp=False,
    ),

    decode_head=dict(
        type='ASPPHead',
        in_channels=64,
        in_index=4,
        channels=16,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        align_corners=False,
        norm_cfg=norm_cfg,

        # Keep original two-class output structure
        num_classes=2,
        out_channels=2,
        ignore_index=255,

        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_name='loss_ce',
                loss_weight=1.0,
                avg_non_ignore=True,
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                naive_dice=True,
                ignore_index=0,
                loss_name='loss_dice',
                loss_weight=3.0,
            ),
        ],
    ),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        align_corners=False,
        norm_cfg=norm_cfg,
        num_classes=2,
        out_channels=2,
        ignore_index=255,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            avg_non_ignore=True,
        ),
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# =============================================================================
# Pipelines
# =============================================================================

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=False,
    ),

    # remap existing 3-class mask only in memory
    dict(
        type='RemapThreeClassToBinary',
        mode='background_vs_candidate',
    ),

    dict(
        type='RandomFlip',
        prob=0.5,
        direction='horizontal',
    ),
    dict(
        type='RandomFlip',
        prob=0.5,
        direction='vertical',
    ),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=False,
    ),
    dict(
        type='RemapThreeClassToBinary',
        mode='background_vs_candidate',
    ),
    dict(type='PackSegInputs'),
]


def make_dataset(split, pipeline):
    return dict(
        type=dataset_type,
        ann_file='',
        metainfo=metainfo,
        reduce_zero_label=False,
        data_root=data_root,

        data_prefix=dict(
            img_path=f'{split}/in',
            seg_map_path=f'{split}/out',
        ),

        img_suffix='_in.png',
        seg_map_suffix='_out.png',
        pipeline=pipeline,
    )


# =============================================================================
# DataLoaders
# train + val ONLY
# =============================================================================

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=True,
    ),

    dataset=make_dataset(
        'train',
        train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),

    dataset=make_dataset(
        'val',
        val_pipeline,
    ),
)

# No test_dataloader
# No test_evaluator
# No test_cfg loop

# =============================================================================
# Validation evaluator
# =============================================================================

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=[
        'mIoU',
        'mDice',
    ],
    nan_to_num=0,
)

# =============================================================================
# Training loop
# =============================================================================

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1,
)

val_cfg = dict(
    type='ValLoop',
)

# =============================================================================
# Optimizer
# Base model differential fine-tuning preserved
# =============================================================================

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',

    clip_grad=dict(
        max_norm=5.0,
        norm_type=2,
    ),

    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
    ),

    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(
                lr_mult=0.1,
            ),
        },
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        begin=0,
        end=1,
        by_epoch=True,
    ),
    dict(
        type='PolyLR',
        begin=1,
        end=20,
        by_epoch=True,
        power=0.9,
        eta_min=1e-4,
    ),
]

# =============================================================================
# Hooks
# =============================================================================

default_hooks = dict(
    timer=dict(
        type='IterTimerHook',
    ),

    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=True,
    ),

    param_scheduler=dict(
        type='ParamSchedulerHook',
    ),

    # save every epoch.
    # Select final checkpoint by foreground validation IoU/Dice.
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        save_last=True,
        max_keep_ckpts=20,
    ),

    sampler_seed=dict(
        type='DistSamplerSeedHook',
    ),

    visualization=dict(
        type='SegVisualizationHook',
        draw=False,
        interval=50,
    ),
)

visualizer = dict(
    _delete_=True,
    type='SegLocalVisualizer',
    name='visualizer',
    vis_backends=[
        dict(
            type='LocalVisBackend',
        ),
    ],
)

log_level = 'INFO'
log_processor = dict(
    by_epoch=True,
)

randomness = dict(
    seed=0,
    deterministic=False,
)

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(
        backend='nccl',
    ),
    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=0,
    ),
)

work_dir = 'projects/filament_segmentation/current_policy/independent_binary/work_dirs/unet_deeplabv3_hrf_split_v2_background_vs_candidate'
