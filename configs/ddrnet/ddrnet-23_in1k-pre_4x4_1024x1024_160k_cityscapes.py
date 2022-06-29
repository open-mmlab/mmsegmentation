_base_ = [
    '../_base_/models/ddrnet.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
 
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])

# model settings
# default : DDRNet23 
norm_cfg = dict(type='SyncBN', eps=1e-03, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='DualResNet',
        layers=[2, 2, (2,), 2],
        planes=64,
        spp_planes=128,
        norm_cfg=norm_cfg ,
        align_corners=False,
        init_cfg= dict(type='Pretrained', checkpoint='/home/yyq/mmseg_ddr23.pth'),
         ),
    decode_head=dict(
        type='FCNHead',
        in_index=-1,
        dropout_ratio=0,
        input_transform=None,
        in_channels=64*4,
        channels=128,
        num_convs=1,
        num_classes=19,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
     auxiliary_head=dict(
        type='FCNHead',
        in_index=-2,
        dropout_ratio=0,
        input_transform=None,
        in_channels=64*2,
        channels=128,
        num_convs=1,
        num_classes=19,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

    # model training and testing settings
    train_cfg=dict(sampler=None),
    test_cfg=dict(mode='whole'))
