# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.1)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    backbone=dict(
        type='HRFormer',
        norm_cfg=norm_cfg,
        norm_eval=False,
        drop_path_rate=0.2,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='HRFORMER',
                window_sizes=(7, 7),
                num_heads=(1, 2),
                mlp_ratios=(4, 4),
                num_blocks=(2, 2),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='HRFORMER',
                window_sizes=(7, 7, 7),
                num_heads=(1, 2, 4),
                mlp_ratios=(4, 4, 4),
                num_blocks=(2, 2, 2),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='HRFORMER',
                window_sizes=(7, 7, 7, 7),
                num_heads=(1, 2, 4, 8),
                mlp_ratios=(4, 4, 4, 4),
                num_blocks=(2, 2, 2, 2),
                num_channels=(32, 64, 128, 256)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[32, 64, 128, 256],
            channels=sum([32, 64, 128, 256]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[32, 64, 128, 256],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
