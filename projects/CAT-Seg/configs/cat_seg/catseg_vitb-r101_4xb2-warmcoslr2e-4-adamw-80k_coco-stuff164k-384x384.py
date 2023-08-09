_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py',
    '../_base_/datasets/coco-stuff164k_384x384.py'
]

custom_imports = dict(imports=['cat_seg'])

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (384, 384)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    # due to the clip model, we do normalization in backbone forward()
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
# model_cfg
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='CLIPOVCATSeg',
        feature_extractor=dict(
            type='ResNet',
            depth=101,
            # only use the first three layers
            num_stages=3,
            out_indices=(0, 1, 2),
            dilations=(1, 1, 1),
            strides=(1, 2, 2),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'),
        ),
        train_class_json='data/coco.json',
        test_class_json='data/coco.json',
        clip_pretrained='ViT-B/16',
        clip_finetune='attention',
    ),
    neck=dict(
        type='CATSegAggregator',
        appearance_guidance_dim=1024,
        num_layers=2,
    ),
    decode_head=dict(
        type='CATSegHead',
        in_channels=128,
        channels=128,
        num_classes=171,
        embed_dims=128,
        decoder_dims=(64, 32),
        decoder_guidance_dims=(512, 256),
        decoder_guidance_proj_dims=(32, 16),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=crop_size, crop_size=crop_size))

# dataset settings
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
)

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=4000))

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.feature_extractor': dict(lr_mult=0.01),
            'backbone.clip_model.visual': dict(lr_mult=0.01)
        }))

# learning policy
param_scheduler = [
    # Use a linear warm-up at [0, 100) iterations
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    # Use a cosine learning rate at [100, 900) iterations
    dict(
        type='CosineAnnealingLR',
        T_max=79500,
        by_epoch=False,
        begin=500,
        end=80000),
]
