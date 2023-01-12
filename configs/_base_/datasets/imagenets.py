# dataset settings
dataset_type = 'ImageNetSDataset'
subset = 919
data_root = 'data/ImageNetS/ImageNetS919'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageNetSImageFromFile', downsample_large_image=True),
    dict(type='LoadImageNetSAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(1024, 256), ratio_range=(0.5, 2.0)),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        cat_max_ratio=0.75,
        ignore_index=1000),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=1000),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageNetSImageFromFile', downsample_large_image=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        subset=subset,
        data_root=data_root,
        img_dir='train-semi',
        ann_dir='train-semi-segmentation',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        subset=subset,
        data_root=data_root,
        img_dir='validation',
        ann_dir='validation-segmentation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        subset=subset,
        data_root=data_root,
        img_dir='validation',
        ann_dir='validation-segmentation',
        pipeline=test_pipeline))
