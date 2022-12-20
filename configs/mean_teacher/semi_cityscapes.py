# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
crop_size = (512, 1024)

branch_field = ['sup', 'unsup']

# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackSegInputs'))
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo masks.
unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
# strong_pipeline = [

# ]

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(
#         type='RandomResize',
#         scale=(2048, 1024),
#         ratio_range=(0.5, 2.0),
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='semi_anns/cityscapes.1@10.json',
    data_prefix=dict(
        img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
    pipeline=sup_pipeline)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='semi_anns/cityscapes.1@10-unlabeled.json',
    data_prefix=dict(
        img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
    pipeline=unsup_pipeline)

train_dataloader = dict(
    batch_size=4,
    num_workers=5,
    sampler=dict(type='MultiSourceSampler', batch_size=4, source_ratio=[2, 2]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
