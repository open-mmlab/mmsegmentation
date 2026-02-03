# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    #mean= [117.926186, 117.568402, 97.217239],
    #std= [53.542876104049824, 50.084170325219176, 50.49331035114637],
   #mean= [432.02181, 315.92948, 246.468659, 310.61462, 360.267789],
   #std= [97.73313111900238, 85.78646917160748, 95.78015824658593, 124.84677067613467, 251.73965882246978],
    mean=[0.23651549, 0.31761484, 0.18514981,   0.26901252, -14.57879175,  -8.6098158,  -14.2907338,  -8.33534564],
    std=[0.16280619, 0.20849304, 0.14008107, 0.19767644, 4.07141682, 3.94773216, 4.21006244, 4.05494136],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    #pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        in_channels=8,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
