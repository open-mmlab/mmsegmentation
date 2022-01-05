# dataset settings
dataset_type = 'VaihingenDataset'
data_root = 'G:/Datasets/Vaihingen/ISPRS_semantic_labeling_Vaihingen'
img_norm_cfg = dict(
    mean=[120.476, 81.7993, 81.1927,30.672], std=[54.8465, 39.3214, 37.9183,38.0866], to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile',color_type='unchanged'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 1536), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='RandomRotate', prob=1, degree=30, auto_bound=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile',color_type='unchanged'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1536),
        #img_scale=None,
        #img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
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
        data_root=data_root,
        img_dir='with_nDSM/tra_val',
        ann_dir='annotations/tra_val',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='with_nDSM/testing',
        ann_dir='annotations/testing',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='with_nDSM/testing',
        ann_dir='annotations/testing',
        pipeline=test_pipeline))

