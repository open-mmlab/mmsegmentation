_base_ = './ocrnet_hrformer-s_4x2_512x1024_80k_cityscapes.py'
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.1)
model = dict(
    backbone=dict(
        drop_path_rate=0.4,
        init_cfg=dict(checkpoint='pretrain/hrt_base.pth'),
        extra=dict(
            stage2=dict(num_heads=(2, 4), num_channels=(78, 156)),
            stage3=dict(num_heads=(2, 4, 8), num_channels=(78, 156, 312)),
            stage4=dict(
                num_heads=(2, 4, 8, 16), num_channels=(78, 156, 312, 624)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[78, 156, 312, 624],
            channels=512,
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=3,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            sampler=dict(type='OHEMPixelSampler', thresh=0.9,
                         min_kept=100000)),
        dict(
            type='OCRHead',
            in_channels=[78, 156, 312, 624],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=100000))
    ])
