# dataset settings
dataset_type = "RoadAnomalyDataset"
data_root = "/misc/lmbraid17/datasets/public/RoadAnomaly/RoadAnomaly_jpg_converted/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (769, 769)
max_ratio = 2

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2049, 1025),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            # dict(type="LoadAnnotations"),
            # dict(type="DefaultFormatBundle"),
            dict(type='ImageToTensor', keys=['img']),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="frames",
        ann_dir="frames",
        img_suffix='.jpg',
        seg_map_suffix='.labels/labels_semantic_converted.png',
        pipeline=test_pipeline
    ),
)
