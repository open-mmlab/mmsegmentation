_base_ = [
    '../_base_/datasets/mounted_empty.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_5k.py'
]

# Note for running this you need to change the model_image_size in
# `configs/_base_/datasets/mounted_empty.py` (1024, 1024)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=(512,512),
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)

norm_cfg = dict(type='SyncBN', requires_grad=True)

model= dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SAMImageEncoderViT_B',
    ),
    # maybe we don't need the neck here
    neck = dict(
        type ='MultiLevelNeck',
        in_channels=[256, 256, 256, 256],
        out_channels=256,
        scales=[4,2,1,0.5],
    ),
    decode_head = dict(
        type='UPerHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)


