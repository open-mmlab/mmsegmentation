# dataset settings
dataset_type = "StreetHazardsDataset"
data_root = "/misc/lmbraid17/datasets/public/StreetHazards/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (769, 769)
max_ratio = 2
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1280, 720), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2049, 1025),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="train/images/training",
        ann_dir="train/annotations/training",
        seg_map_suffix='.png',
        img_suffix='.png',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="train/images/validation",
        ann_dir="train/annotations/validation",
        seg_map_suffix='.png',
        img_suffix='.png',
        pipeline=test_pipeline,

    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="test/images/test",
        ann_dir="test/annotations/test",
        seg_map_suffix='.png',
        img_suffix='.png',
        pipeline=test_pipeline,
    ),
)
