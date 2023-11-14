
data_root = "/data/dataset"
dataset_type = "MountedEmpty"
#model_image_size = (1024, 1024) # for SAM
model_image_size = (512, 512)
keep_ratio = False
reduce_zero_label=False

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=model_image_size, keep_ratio=keep_ratio),
    dict(type='RandomFlip', prob=0.5, direction = "horizontal"),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=model_image_size, keep_ratio=keep_ratio),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=model_image_size , keep_ratio=keep_ratio)
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=reduce_zero_label,
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=reduce_zero_label,
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=reduce_zero_label,
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        pipeline=test_pipeline))

train_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'], prefix="test")
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'], prefix="val")
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'], prefix="test")

