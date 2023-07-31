# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='MobileSeg_Base',
        channels=[16, 32, 64, 128, 192],
        depths=[3, 3],
        embed_dims=[128, 192],
        num_heads=8,
        inj_type='AAMSx8',
        out_feat_chs=[64, 128, 192],
    ),
    decode_head=dict(
        type='PPMobileSegHead',
        num_classes=150,
        in_channels=256,
        dropout_ratio=0.1,
        use_dw=True,
        align_corners=False),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
