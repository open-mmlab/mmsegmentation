# UNet for GF 5-channel imagery
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    #mean=[117.926186, 117.568402, 97.217239],
    #std=[53.542876104049824, 50.084170325219176, 50.49331035114637],
    #mean= [432.02181, 315.92948, 246.468659, 310.61462, 360.267789],
    #std= [97.73313111900238, 85.78646917160748, 95.78015824658593, 124.84677067613467, 251.73965882246978],
    mean=[0.23651549, 0.31761484, 0.18514981,   0.26901252, -14.57879175,  -8.6098158,  -14.2907338,  -8.33534564],
    std=[0.16280619, 0.20849304, 0.14008107, 0.19767644, 4.07141682, 3.94773216, 4.21006244, 4.05494136],
    bgr_to_rgb=False,  # Not RGB imagery
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=8,  # 5-channel input
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='ASPPHead',
        in_channels=64,
        in_index=4,
        channels=16,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=256, stride=170))
