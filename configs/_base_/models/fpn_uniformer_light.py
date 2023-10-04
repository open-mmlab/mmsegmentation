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
    pretrained=None,
    backbone=dict(
        type='UniFormer_Light',
        depth=[3, 5, 9, 3],
        conv_stem=True,
        prune_ratio=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5]],
        trade_off=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5]],
        embed_dim=[64, 128, 256, 512],
        head_dim=32,
        mlp_ratio=[3, 3, 3, 3],
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
