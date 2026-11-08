"""MMSegmentation 1.x config for the final 5-region 3-class dataset.

Dataset layout:
    /mnt/sp4t/fujiwara.r/split/
      train/in/{region}/{chunk}/*_in.png
      train/out/{region}/{chunk}/*_out.png
      val/in/{region}/{chunk}/*_in.png
      val/out/{region}/{chunk}/*_out.png
      test/in/{region}/{chunk}/*_in.png
      test/out/{region}/{chunk}/*_out.png

Labels:
    0   = background
    1   = confirmed filament
    2   = candidate filament
    255 = ignore

The HRF-pretrained DeepLabV3 with UNet backbone, its checkpoint path, backbone
definition, ASPP decoder structure and auxiliary head structure are preserved.
Only the dataset, 3-class output, training loop and runtime settings are adapted
to the generated astronomical patches.
"""

_base_ = [
    '../../../../configs/_base_/default_runtime.py',
]

default_scope = 'mmseg'
custom_imports = dict(
    imports=['mmseg.models', 'foreground_balanced_dataset'],
    allow_failed_imports=False,
)

# =============================================================================
# Dataset
# =============================================================================
dataset_type = 'BaseSegDataset'
data_root = '/mnt/sp4t/fujiwara.r/split'
crop_size = (256, 256)
reduce_zero_label = False

metainfo = dict(
    classes=(
        'background',
        'confirmed_filament',
        'candidate_filament',
    ),
    palette=[
        [0, 0, 0],
        [255, 220, 0],
        [0, 150, 255],
    ],
)

# Approximate train pixel distribution from the generated dataset:
#   background          96.47%
#   confirmed filament   1.71%
#   candidate filament   1.82%
#
# The previous 1:6:6 ratio still left about 82% of weighted CE mass on
# background. Use 0.1:3:3 (relative ratio 1:30:30), which makes the expected
# weighted contributions approximately background 48%, class 1 25% and
# class 2 27%.
class_weight = [0.1, 3.0, 3.0]

# =============================================================================
# HRF-pretrained DeepLabV3 + UNet checkpoint: unchanged
# =============================================================================
load_from = (
    'checkpoints/'
    'deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_'
    '256x256_40k_hrf_20211210_202032-59daf7a4.pth'
)
resume = False

# Same preprocessing used by the HRF-pretrained model.
data_preprocessor = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
)

norm_cfg = dict(type='BN', requires_grad=True)

# =============================================================================
# Model: DeepLabV3 decoder + UNet backbone
# Architecture is preserved from the supplied HRF-pretrained configuration.
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

        # Only the classifier output is adapted from HRF's 2 classes to
        # background / confirmed filament / candidate filament.
        num_classes=3,
        out_channels=3,
        ignore_index=255,

        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_name='loss_ce',
                loss_weight=1.0,
                class_weight=class_weight,
                avg_non_ignore=True,
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                naive_dice=True,
                # Dice optimizes the two filament channels. Cross entropy still
                # trains all three classes, including background.
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
        num_classes=3,
        out_channels=3,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=class_weight,
            avg_non_ignore=True,
        ),
    ),

    train_cfg=dict(),

    # Every input is already one complete 256x256 patch.
    test_cfg=dict(mode='whole'),
)

# =============================================================================
# Pipelines
# =============================================================================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackSegInputs'),
]

val_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]


def make_dataset(split):
    """Read all region/chunk subdirectories below one split recursively."""
    dataset = dict(
        type=(
            'ForegroundBalancedDataset'
            if split == 'train'
            else dataset_type
        ),
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
        pipeline=(
            train_pipeline
            if split == 'train'
            else val_test_pipeline
        ),
    )

    if split == 'train':
        dataset.update(
            class1_repeat=6,
            class2_repeat=3,
            min_class1_pixels=16,
            min_class2_pixels=16,
            balance_seed=0,
        )

    return dataset


# Generated dataset:
#   train = 675 patches
#   val   = 272 patches
#   test  = 179 patches
#
# batch_size=8 gives ceil(675/8)=85 optimizer updates per epoch.
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True,
    ),
    dataset=make_dataset('train'),
)

# Evaluation has no backward pass, so batch 8 is safe if train batch 8 fits.
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),
    dataset=make_dataset('val'),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),
    dataset=make_dataset('test'),
)

# =============================================================================
# Evaluation
# =============================================================================
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    nan_to_num=0,
)

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    nan_to_num=0,
)

# =============================================================================
# Training loop
# =============================================================================
# ForegroundBalancedDataset increases the effective train length by
# repeating only positive patches. Ten balanced epochs provide substantially
# more foreground-bearing batches than twenty epochs of the original list.
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=1,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# =============================================================================
# Optimization
# =============================================================================
# Preserve differential fine-tuning:
#   HRF-pretrained UNet backbone: 0.001
#   segmentation heads:          0.01
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
            'backbone': dict(lr_mult=0.1),
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
        end=10,
        by_epoch=True,
        power=0.9,
        eta_min=1e-4,
    ),
]

# =============================================================================
# Hooks
# =============================================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=True,
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        save_last=True,
        max_keep_ckpts=5,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
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
        dict(type='LocalVisBackend'),
    ],
)

# =============================================================================
# Runtime
# =============================================================================
log_level = 'INFO'
log_processor = dict(by_epoch=True)
randomness = dict(
    seed=0,
    deterministic=False,
)

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=0,
    ),
)

work_dir = (
    'projects/filament_segmentation/transition/work_dirs/'
    'unet_deeplabv3_hrf_3class_5regions_lshape_balanced_v2'
)


