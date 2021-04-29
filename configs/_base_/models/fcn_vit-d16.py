# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='https://github.com/rwightman/pytorch-image-models/releases/\
download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False),
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[768],
        out_channels=768,
        scales=[0.5, 1, 2, 4]),
    decode_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=1,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
