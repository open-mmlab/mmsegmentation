# dataset settings
dataset_type = 'DarkZurichDataset'
data_root = 'data/dark_zurich/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1080, 960)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1080, 1920),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_anon/val/night/GOPR0356',
        ann_dir='gt/val/night/GOPR0356',
        pipeline=test_pipeline))
