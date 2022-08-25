# model settings
_base_ = './fastfcn_r50-d32_jpu_psp_512x512_80k_ade20k.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        _delete_=True,
        type='ASPPHead',
        in_channels=2048,
        in_index=2,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
