# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ICNet',
        in_channels=3,
        layer_channels=(512, 2048),
        light_branch_middle_channels=32,
        psp_out_channels=512,
        out_channels=(64, 256, 256),
        resnet_cfg=dict(
            type='ResNetV1c',
            in_channels=3,
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True),
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    decode_head=dict(
        type='ICHead',
        in_channels=(64, 256, 256),
        in_index=(0, 1, 2),
        input_transform='multiple_select',
        channels=128,
        dropout_ratio=0,
        norm_cfg=norm_cfg,
        # act_cfg=None, # ablation with act_cfg=None
        align_corners=False,
        num_classes=19,
        # ablation with OHEM
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
