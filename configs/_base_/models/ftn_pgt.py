# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='PyramidGroupTransformer',
        pretrain_img_size=224,
        embed_dims=64,
        patch_size=4,
        group_nums=[64, 16, 1, 1],
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        strides=(4, 2, 2, 2),
        out_indices=(1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512],
        out_channels=512,
        num_outs=3),
    decode_head=dict(
        type='FeaturePyramidTransformerHead',
        in_index=[0, 1, 2],
        in_channels=[512, 512, 512],
        channels=512,
        num_layers=((1, 1, 1), (1, 1), (1)),
        num_heads=4,
        sra_ratios=((2, 2, 2), (2, 2), (2)), 
        mlp_ratio=4,
        num_fcs=2,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=(0.3, 0.2, 0.1),
        use_ape=False,
        ape_sizes=[(16, 16), (32, 32), (64, 64), (128, 128)],
        num_classes=150,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=512,
    #     in_index=1,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=150,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)
    
