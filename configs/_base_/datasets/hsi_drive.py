train_pipeline = [
    dict(type='LoadImageFromNpyFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=(192, 384)),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromNpyFile'),
    dict(type='RandomCrop', crop_size=(192, 384)),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='HSIDrive20Dataset',
        data_root='data/HSIDrive20',
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='HSIDrive20Dataset',
        data_root='data/HSIDrive20',
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='HSIDrive20Dataset',
        data_root='data/HSIDrive20',
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=0)
test_evaluator = val_evaluator
