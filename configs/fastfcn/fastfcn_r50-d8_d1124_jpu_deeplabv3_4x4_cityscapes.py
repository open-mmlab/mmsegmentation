# model settings
_base_ = './fastfcn_r50-d8_d1124_jpu_psp_4x4_cityscapes.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
    ),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
