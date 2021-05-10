# model settings
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
aux_alpha = dict(
    type='SETRMLAAUXHead',
    in_channels=256,
    channels=512,
    in_index=0,
    img_size=(768, 768),
    mla_channels=256,
    num_classes=19,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_beta = dict(
    type='SETRMLAAUXHead',
    in_channels=256,
    channels=512,
    in_index=1,
    img_size=(768, 768),
    mla_channels=256,
    num_classes=19,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_gamma = dict(
    type='SETRMLAAUXHead',
    in_channels=256,
    channels=512,
    in_index=2,
    img_size=(768, 768),
    mla_channels=256,
    num_classes=19,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_delta = dict(
    type='SETRMLAAUXHead',
    in_channels=256,
    channels=512,
    in_index=3,
    img_size=(768, 768),
    mla_channels=256,
    num_classes=19,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
model = dict(
    type='EncoderDecoder',
    pretrained='https://github.com/rwightman/pytorch-image-models/releases/\
download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(768, 768),
        patch_size=16,
        in_channels=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        out_indices=(5, 11, 17, 23),
        drop_rate=0.1,
        norm_cfg=backbone_norm_cfg,
        out_shape='NCHW',
        with_cls_token=False,
        interpolate_mode='bilinear',
    ),
    neck=dict(
        type='MLA',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
    ),
    decode_head=dict(
        type='SETRMLAHead',
        in_channels=(1024, 1024, 1024, 1024),
        channels=512,
        in_index=(0, 1, 2, 3),
        img_size=(768, 768),
        mla_channels=256,
        mlahead_channels=128,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[aux_alpha, aux_beta, aux_gamma, aux_delta],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
