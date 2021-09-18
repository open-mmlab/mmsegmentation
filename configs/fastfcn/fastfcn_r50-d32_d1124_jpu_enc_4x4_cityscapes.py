# model settings
_base_ = './fastfcn_r50-d8_d1124_jpu_psp_4x4_cityscapes.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 2, 2),
    ),
    decode_head=dict(
        type='EncHead',
        in_channels=[512, 1024, 2048],
        in_index=(1, 2, 3),
        channels=512,
        num_codes=32,
        use_se_loss=True,
        add_lateral=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_se_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,)