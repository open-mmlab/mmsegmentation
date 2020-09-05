# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-05)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain_model/resnet50-deep.pth',
    backbone=dict(
        type='ICNet',
        resnet_cfg=dict(
            depth=50,
            stem_channels=128,
            strides=(1, 2, 1, 1),
            dilations=(1, 1, 2, 4),
            deep_stem=True,
            style='pytorch',
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='ICHead',
        in_channels=128,
        channels=128,
        dropout_ratio=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=None,
        num_classes=19,
        in_index=-1,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)))
