norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='STDCContextPathNet',
        stdc_cfg=dict(
            type='StdcNet',
            stdc_type='STDCNet813',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            stdc_num_convs=4,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            with_final_conv=True),
        in_channels=(512, 1024),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4)),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=4,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
