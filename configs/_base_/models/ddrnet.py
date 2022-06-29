# model settings
# default : DDRNet23slim

norm_cfg = dict(type='SyncBN', eps=1e-03, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='DualResNet',
        layers=[2, 2, (2,), 2],
        planes=32,
        spp_planes=128,
        norm_cfg=norm_cfg ,
        align_corners=False,
        init_cfg= dict(type='Pretrained', checkpoint='mmseg_ddr23s.pth'),
         ),
  
    decode_head=dict(
        type='FCNHead',
        init_cfg= dict(type='Kaiming', distribution='normal'),
        in_index=-1,
         concat_input=False,
        dropout_ratio=0,
        input_transform=None,
        in_channels=32*4,
        channels=64,
        num_convs=1,
        num_classes=19,
        align_corners=False,
        norm_cfg=norm_cfg ,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
     auxiliary_head=dict(
        type='FCNHead',
        init_cfg= dict(type='Kaiming', distribution='normal'),
        in_index=-2,
        concat_input=False,
        dropout_ratio=0,
        input_transform=None,
        in_channels=32*2,
        channels=64,
        num_convs=1,
        num_classes=19,
        align_corners=False,
        norm_cfg=norm_cfg ,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

    # model training and testing settings
    train_cfg=dict(sampler=None),
    test_cfg=dict(mode='whole'))
