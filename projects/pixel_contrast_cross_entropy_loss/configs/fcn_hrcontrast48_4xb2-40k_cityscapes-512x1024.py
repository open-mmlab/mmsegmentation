_base_ = './fcn_hrcontrast18_4xb2-40k_cityscapes-512x1024.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        type='ContrastHead',
        in_channels=[48, 96, 192, 384],
        channels=sum([48, 96, 192, 384]),
        proj_n=720,
        seg_head=dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            in_index=(0, 1, 2, 3),
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False),
    ))
