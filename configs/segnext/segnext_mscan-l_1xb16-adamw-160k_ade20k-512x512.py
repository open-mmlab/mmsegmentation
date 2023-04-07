_base_ = './segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py'
# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_l_20230227-cef260d4.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 27, 3],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        drop_path_rate=0.3,
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=1024,
        ham_channels=1024,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
