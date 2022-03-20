# model settings
_base_ = './fastfcn_r50-d32_jpu_psp_512x512_80k_ade20k.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        _delete_=True,
        type='EncHead',
        in_channels=[512, 1024, 2048],
        in_index=(0, 1, 2),
        channels=512,
        num_codes=32,
        use_se_loss=True,
        add_lateral=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_se_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
