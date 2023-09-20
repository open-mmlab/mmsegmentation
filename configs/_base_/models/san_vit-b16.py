# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[122.7709, 116.7460, 104.0937],
    std=[68.5005, 66.6322, 70.3232],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=640,
    test_cfg=dict(size_divisor=32))

num_classes = 171
model = dict(
    type='MultimodalEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/clip_vit_base_patch16_224.pth',
    asymetric_input=True,
    encoder_resolution=0.5,
    image_encoder=dict(
        type='VisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        patch_pad=0,
        in_channels=3,
        embed_dims=768,
        num_layers=9,
        num_heads=12,
        mlp_ratio=4,
        out_origin=True,
        out_indices=(2, 5, 8),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=True,
        patch_bias=False,
        pre_norm=True,
        norm_cfg=dict(type='LN', eps=1e-5),
        act_cfg=dict(type='QuickGELU'),
        norm_eval=False,
        interpolate_mode='bicubic',
        frozen_exclude=['pos_embed']),
    text_encoder=dict(
        type='CLIPTextEncoder',
        dataset_name=None,
        templates='vild',
        embed_dims=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4,
        output_dims=512,
        cache_feature=True,
        cat_bg=True,
        norm_cfg=dict(type='LN', eps=1e-5)
        ),
    decode_head=dict(
        type='SideAdapterCLIPHead',
        num_classes=num_classes,
        deep_supervision_idxs=[7],
        san_cfg=dict(
            in_channels=3,
            clip_channels=768,
            embed_dims=240,
            patch_size=16,
            patch_bias=True,
            num_queries=100,
            cfg_encoder=dict(
                num_encode_layer=8,
                num_heads=6,
                mlp_ratio=4
            ),
            fusion_index=[0, 1, 2, 3],
            cfg_decoder=dict(
                num_heads=12,
                num_layers=1,
                embed_channels=256,
                mlp_channels=256,
                num_mlp=3,
                rescale=True),
            norm_cfg=dict(type='LN', eps=1e-6),
        ),
        maskgen_cfg=dict(
            sos_token_format='cls_token',
            sos_token_num=100,
            cross_attn=False,
            num_layers=3,
            embed_dims=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            out_dims=512,
            final_norm=True,
            act_cfg=dict(type='QuickGELU'),
            norm_cfg=dict(type='LN', eps=1e-5),
            frozen_exclude=[]
        ),
        align_corners=False,
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='ClassificationCost', weight=2.0),
                    dict(
                        type='CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ])),
        loss_decode=[dict(type='CrossEntropyLoss',
                          loss_name='loss_cls_ce',
                          loss_weight=2.0,
                          class_weight=[1.0] * num_classes + [0.1]),
                     dict(type='CrossEntropyLoss',
                          use_sigmoid=True,
                          loss_name='loss_mask_ce',
                          loss_weight=5.0),
                     dict(type='DiceLoss',
                          ignore_index=None,
                          naive_dice=True,
                          eps=1,
                          loss_name='loss_mask_dice',
                          loss_weight=5.0)
                     ]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
