_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/hsi_drive.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
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
