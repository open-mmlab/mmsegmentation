# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain_model/resnet50c128_torchcv-6e57a75d.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        deep_stem=True,
        base_channels=128,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ANNHead',
        in_channels=[1024, 2048],
        channels=512,
        project_channels=256,
        drop_out_ratio=0.05,
        norm_cfg=norm_cfg,
        num_classes=19,
        in_index=[-2, -1],
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        channels=512,
        num_convs=1,
        num_classes=19,
        drop_out_ratio=0.05,
        in_index=-2,
        norm_cfg=norm_cfg,
        concat_input=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
