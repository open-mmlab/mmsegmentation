# dataset settings
dataset_type = 'REFUGEDataset'
data_root = 'C:/OpenMMlab/mmsegmentation/data/REFUGE'
train_img_scale = (2056, 2124)
val_img_scale = (1634, 1634)
test_img_scale = (1634, 1634)
crop_size = (256, 256)
train_pipeline = [dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=train_img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFliplr', prob=0.2),

    dict(type='RandomFlipud', prob=0.2),
    dict(type='RandomApply',
        transforms=[dict(type='Rot90', degree_range=(1,3))],
    prob=0.3),
    # dict(type='PhotoMetricDistortion'),
    dict(type='RandomChoice',
        transforms=[dict(type='ColorJitter', brightness=0.2),
        dict(type='ColorJitter', contrast=0.2),
        dict(type='ColorJitter', saturation=0.2),
        dict(type='ColorJitter', brightness=0.1, contrast=0.1, saturation=0.1, hue=0)]),
    # dict(type='ToTensor'),
    dict(type='PackSegInputs')]

    
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=val_img_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=test_img_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='images/training',
                seg_map_path='annotations/training'),
            pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='annotations/test'),
        pipeline=val_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator
