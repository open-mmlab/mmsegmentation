# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='https://github.com/rwightman/pytorch-image-models/releases/\
download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(768, 768),
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        out_indices=(5, 11, 17, 23),
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        with_cls_token=False,
        align_corners=False,
        interpolate_mode='bilinear',
    ),
    decode_head=dict(
        type='SETRMLAHead',
        in_channels=(1024, 1024, 1024, 1024),
        channels=512,
        in_index=(0, 1, 2, 3),
        img_size=(768, 768),
        embed_dim=1024,
        mla_channels=256,
        mlahead_channels=128,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='SETRMLAAUXHead',
            in_channels=(1024, 1024, 1024, 1024),
            channels=512,
            in_index=(0, 1, 2, 3),
            img_size=(768, 768),
            embed_dim=1024,
            mla_channels=256,
            mla_select_index=0,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRMLAAUXHead',
            in_channels=(1024, 1024, 1024, 1024),
            channels=512,
            in_index=(0, 1, 2, 3),
            img_size=(768, 768),
            embed_dim=1024,
            mla_channels=256,
            mla_select_index=1,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRMLAAUXHead',
            in_channels=(1024, 1024, 1024, 1024),
            channels=512,
            in_index=(0, 1, 2, 3),
            img_size=(768, 768),
            embed_dim=1024,
            mla_channels=256,
            mla_select_index=2,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRMLAAUXHead',
            in_channels=(1024, 1024, 1024, 1024),
            channels=512,
            in_index=(0, 1, 2, 3),
            img_size=(768, 768),
            embed_dim=1024,
            mla_channels=256,
            mla_select_index=3,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ])
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
