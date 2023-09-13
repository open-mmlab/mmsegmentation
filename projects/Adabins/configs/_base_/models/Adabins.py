# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='DepthEstimator',
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='AdabinsBackbone',
        basemodel_name='tf_efficientnet_b5_ap',
        num_features=2048,
        num_classes=128,
        bottleneck_features=2048,
    ),
    decode_head=dict(
        type='AdabinsHead',
        in_channels=128,
        n_query_channels=128,
        patch_size=16,
        embedding_dim=128,
        num_heads=4,
        n_bins=256,
        min_val=0.001,
        max_val=10,
        norm='linear'),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
