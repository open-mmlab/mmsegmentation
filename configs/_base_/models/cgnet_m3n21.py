# model settings
norm_cfg = dict(type='SyncBN', eps=1e-03, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='CGNet',
        norm_cfg=norm_cfg,
        in_channels=3,
        num_channels=[32, 64, 128],
        num_blocks=[3, 21],
        dilation=[2, 4],
        reduction=[8, 16]),
    decode_head=dict(
        type='CGHead',
        in_channels=256,
        channels=256,
        norm_cfg=norm_cfg,
        num_classes=19,
        in_index=-1,
        dropout_ratio=0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
