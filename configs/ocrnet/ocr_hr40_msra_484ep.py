_base_ = './ocr_hr18_msra_484ep.py'
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w40',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(40, 80)),
            stage3=dict(num_channels=(40, 80, 160)),
            stage4=dict(num_channels=(40, 80, 160, 320)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[40, 80, 160, 320],
            channels=sum([40, 80, 160, 320]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            drop_out_ratio=-1,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[40, 80, 160, 320],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            drop_out_ratio=-1,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])
