# model settings
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=\
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',  # noqa
    backbone=dict(
        type='VisionTransformer',
        img_size=(768, 768),
        patch_size=16,
        in_channels=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        out_indices=(9, 14, 19, 23),
        drop_rate=0.1,
        norm_cfg=backbone_norm_cfg,
        out_shape='NCHW',
        with_cls_token=True,
        interpolate_mode='bilinear',
    ),
    decode_head=dict(
        type='SETRUPHead',
        in_channels=1024,
        channels=512,
        in_index=3,
        embed_dim=1024,
        num_classes=19,
        norm_cfg=norm_cfg,
        num_convs=4,
        up_mode='bilinear',
        num_up_layer=4,
        kernel_size=3,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=0,
            embed_dim=1024,
            num_classes=19,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=1,
            embed_dim=1024,
            num_classes=19,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=2,
            embed_dim=1024,
            num_classes=19,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=3,
            embed_dim=1024,
            num_classes=19,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
