# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='CCHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        recurrence=2,
        drop_out_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        drop_out_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
