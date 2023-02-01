# model settings
norm_cfg = dict(type='SyncBN', eps=1e-03, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[72.39239876, 82.90891754, 73.15835921],
    std=[1, 1, 1],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='CGNet',
        norm_cfg=norm_cfg,
        in_channels=3,
        num_channels=(32, 64, 128),
        num_blocks=(3, 21),
        dilations=(2, 4),
        reductions=(8, 16)),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0,
        num_classes=19,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[
                2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352,
                10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905,
                10.347791, 6.3927646, 10.226669, 10.241062, 10.280587,
                10.396974, 10.055647
            ])),
    # model training and testing settings
    train_cfg=dict(sampler=None),
    test_cfg=dict(mode='whole'))
