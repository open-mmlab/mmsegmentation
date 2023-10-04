_base_ = [
    '../../../configs/_base_/models/fcn_unet_s5-d16.py',
    './_base_/datasets/hsi_drive.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_160k.py'
]

custom_imports = dict(
    imports=['projects.hsidrive20_dataset.mmseg.datasets.hsi_drive'])

crop_size = (192, 384)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=None,
    std=None,
    bgr_to_rgb=None,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(in_channels=25),
    decode_head=dict(
        ignore_index=0,
        num_classes=11,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True)),
    auxiliary_head=dict(
        ignore_index=0,
        num_classes=11,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
