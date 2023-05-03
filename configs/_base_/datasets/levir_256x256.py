# dataset settings
dataset_type = 'LEVIRCDDataset'
data_root = r'data/LEVIRCD'
crop_size = (256, 256)

albu_train_transforms = [
    dict(type='RandomRotate90', p=1),
    dict(type='RandomBrightnessContrast', p=0.2),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5)
]

train_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Albu', transforms=albu_train_transforms),
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[[dict(type='LoadAnnotations')],
                    [dict(type='ConcatCDInput')],
                    [dict(type='PackSegInputs')]])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='list/train.txt',
        data_root=data_root,
        data_prefix=dict(img_path='A', img_path2='B', seg_map_path='label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='list/val.txt',
        data_root=data_root,
        data_prefix=dict(img_path='A', img_path2='B', seg_map_path='label'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
