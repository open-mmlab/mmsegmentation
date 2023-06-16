dataset_type = 'FaceOccludedDataset'
data_root = 'data/occlusion-aware-face-dataset'
crop_size = (512, 512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', degree=(-30, 30), prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_train_A = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='NatOcc_hand_sot/img',
    ann_dir='NatOcc_hand_sot/mask',
    split='train.txt',
    pipeline=train_pipeline)

dataset_train_B = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='NatOcc_object/img',
    ann_dir='NatOcc_object/mask',
    split='train.txt',
    pipeline=train_pipeline)

dataset_train_C = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='RandOcc/img',
    ann_dir='RandOcc/mask',
    split='train.txt',
    pipeline=train_pipeline)

dataset_valid = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='RealOcc/image',
    ann_dir='RealOcc/mask',
    split='RealOcc/split/val.txt',
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[dataset_train_A, dataset_train_B, dataset_train_C],
    val=dataset_valid)
