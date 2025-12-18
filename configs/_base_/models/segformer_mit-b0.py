# model settings 这实际上是b5的参数量
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    #mean=[117.926186, 117.568402, 97.217239],
    #std=[53.542876104049824, 50.084170325219176, 50.49331035114637],
    mean= [432.02181, 315.92948, 246.468659, 310.61462, 360.267789],
    std= [97.73313111900238, 85.78646917160748, 95.78015824658593, 124.84677067613467, 251.73965882246978],
    #mean=[0.23651549, 0.31761484, 0.18514981,   0.26901252, -14.57879175,  -8.6098158,  -14.2907338,  -8.33534564],
    #std=[0.16280619, 0.20849304, 0.14008107, 0.19767644, 4.07141682, 3.94773216, 4.21006244, 4.05494136],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=5,
        embed_dims=64,  # B0: 32 -> B5: 64
        num_stages=4,
        num_layers=[3, 6, 40, 3],  # B0: [2, 2, 2, 2] -> B5: [3, 6, 40, 3]
        num_heads=[1, 2, 5, 8],  # 保持不变
        patch_sizes=[7, 3, 3, 3],  # 保持不变
        sr_ratios=[8, 4, 2, 1],  # 保持不变
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],  # B0: [32, 64, 160, 256] -> B5: [64, 128, 320, 512]
        in_index=[0, 1, 2, 3],
        channels=768,  # B0: 256 -> B5: 768
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
