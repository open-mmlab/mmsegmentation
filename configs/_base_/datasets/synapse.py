dataset_type = "SynapseDataset"
data_root = 'data/synapse/'
img_scale = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomChoice',
         transforms=[
            [dict(type='RandomRotate', prob=1, degree=[-20, 20])],
            [
                dict(type='RandomChoice',
                     transforms=[
                         [dict(type='RandomRotate', prob=1., degree=[0, 0])],
                         [dict(type='RandomRotate', prob=1., degree=[90, 90])],
                         [dict(type='RandomRotate', prob=1., degree=[180, 180])],
                         [dict(type='RandomRotate', prob=1., degree=[-90, -90])]
                     ]),
                dict(type='RandomFlip', prob=1.)
            ]
         ]),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
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
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
