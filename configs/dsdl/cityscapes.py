_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
# dataset settings
dataset_type = 'DSDLSegDataset'
data_root = 'data/CityScapes'
img_prefix = 'raw/CityScapes'
train_ann = 'dsdl/dsdl_SemSeg_full/set-train/train.yaml'
val_ann = 'dsdl/dsdl_SemSeg_full/set-val/val.yaml'

used_labels = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
    'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

train_pipeline = [
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
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
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
        data_prefix=dict(img_path=img_prefix, seg_map_path=img_prefix),
        ann_file=train_ann,
        used_labels=used_labels,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_prefix, seg_map_path=img_prefix),
        ann_file=val_ann,
        used_labels=used_labels,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
