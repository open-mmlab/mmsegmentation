# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='HRViT',
        channels=64,
        channel_list=(
            (32, ),
            (32, 64),
            (32, 64, 128),
            (32, 64, 128),
            (32, 64, 128, 256),
            (32, 64, 128, 256),
        ),
        block_list=(
            (1, ),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
            (1, 1, 6, 2),
            (1, 1, 2, 2),
        ),
        dim_head=32,
        ws_list=(1, 2, 7, 7),
        proj_dropout=0.0,
        mlp_ratio_list=(4, 4, 4, 4),
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
