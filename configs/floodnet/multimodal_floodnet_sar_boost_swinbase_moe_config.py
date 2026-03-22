"""
Swin-Base + MoE: Multi-Dataset FloodNet Training - SAR-Boosted Configuration
MMSeg 1.x Version

Based on Swin-uavflood-256x256 (Swin-Base) upgraded with MoE.
Backbone: Swin-Base (embed_dims=128) + MoE (8 experts, top_k=3)
Estimated params: ~456M (vs Swin-T+MoE ~278M, vs Swin-B ~109M)

Usage:
    python tools/train.py configs/floodnet/multimodal_floodnet_sar_boost_swinbase_moe_config.py \
        --work-dir work_dirs/swinbase_moe_upernet/SAR_Boost/ --seed 42
"""

_base_ = [
    '../_base_/default_runtime.py',
]

# ==================== Dataset Definition ====================
DATASETS_CONFIG = dict(
    names=['sar', 'rgb', 'GF'],
)

# ==================== Modal Definition ====================
ALL_KNOWN_MODALS = {
    'sar': {'channels': 8, 'pattern': 'sar',
            'description': 'Synthetic Aperture Radar'},
    'rgb': {'channels': 3, 'pattern': 'rgb',
            'description': 'RGB Optical'},
    'GF': {'channels': 5, 'pattern': 'GF',
           'description': 'GaoFen Satellite'},
}

TRAINING_MODALS = ['sar', 'rgb', 'GF']
num_classes = 2

# ==================== MoE Config ====================
depths = [2, 2, 18, 2]
MoE_Block_inds = [
    [],
    [1],
    [1, 3, 5, 7, 9, 11, 13, 15, 17],
    [0, 1]
]

num_shared_experts_config = {
    0: 0,
    1: 0,
    2: 2,
    3: 1
}

num_experts = 8
top_k = 3
noisy_gating = True

# ==================== Model Config ====================
norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (256, 256)

data_preprocessor = dict(
    type='MultiModalDataPreProcessor',
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

model = dict(
    type='MultiModalEncoderDecoderV2',
    data_preprocessor=data_preprocessor,
    use_moe=True,
    use_modal_bias=True,
    moe_balance_weight=1.0,
    moe_diversity_weight=0.1,
    multi_tasks_reweight=None,
    decoder_mode='separate',
    dataset_names=DATASETS_CONFIG['names'],

    backbone=dict(
        type='MultiModalSwinMoE',
        modal_configs=ALL_KNOWN_MODALS,
        training_modals=TRAINING_MODALS,

        pretrain_img_size=224,
        patch_size=4,
        # ---- Swin-Base dimensions ----
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,  # Swin-B uses 0.3 (vs Swin-T 0.2)
        patch_norm=True,
        out_indices=[0, 1, 2, 3],

        # ---- MoE ----
        use_moe=True,
        num_experts=num_experts,
        num_shared_experts_config=num_shared_experts_config,
        top_k=top_k,
        noisy_gating=noisy_gating,
        MoE_Block_inds=MoE_Block_inds,
        use_expert_diversity_loss=True,

        pretrained=None,
    ),

    # ---- Swin-Base output channels: [128, 256, 512, 1024] ----
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,  # Swin-B stage2 output
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        )
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(170, 170))
)

# ==================== Dataset Config ====================
dataset_type = 'MultiModalDeepflood'
data_root = '../floodnet/data/mixed_dataset/'

train_pipeline = [
    dict(type='LoadMultiModalImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='GenerateBoundary', thickness=3),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MultiModalNormalize'),
    dict(type='MultiModalPad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PackMultiModalSegInputs',
         meta_keys=('img_path', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'modal_type', 'actual_channels', 'dataset_name',
                    'img_norm_cfg', 'reduce_zero_label')),
]

test_pipeline = [
    dict(type='LoadMultiModalImageFromFile', to_float32=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='MultiModalNormalize'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackMultiModalSegInputs',
         meta_keys=('img_path', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'modal_type', 'actual_channels', 'dataset_name',
                    'img_norm_cfg', 'reduce_zero_label')),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        type='FixedRatioModalSampler',
        modal_ratios={'sar': 6, 'rgb': 5, 'GF': 5},
        modal_order=['sar', 'rgb', 'GF'],
        reference_modal='GF',
        batch_size=16,
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/labels'),
        pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/labels'),
        pipeline=test_pipeline),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/images',
            seg_map_path='test/labels'),
        pipeline=test_pipeline),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ==================== Optimizer Config ====================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'patch_embed': dict(lr_mult=2.0),
            'modal_patch_embeds': dict(lr_mult=2.0),
            'relative_position_bias_table': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'gating': dict(lr_mult=2.0),
            'experts': dict(lr_mult=1.5),
            'modal_bias': dict(lr_mult=3.0),
            'shared_experts': dict(lr_mult=2.0),
        }
    )
)

# ==================== Learning Rate Config ====================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=5),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=5,
        end=100,
        by_epoch=True),
]

# ==================== Training Loop ====================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ==================== Hooks ====================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=500, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=10,
        max_keep_ckpts=1,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

# ==================== Runtime Overrides ====================
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(by_epoch=True)
