# model settings
norm_cfg = dict(type='NaiveSyncBN', requires_grad=True)
model = dict(
    type='EncodeDecode',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        channels=512,
        norm_cfg=norm_cfg,
        num_classes=19,
        in_index=-1,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        channels=512,
        num_convs=1,
        num_classes=19,
        in_index=-2,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
