_base_ = [
    '../_base_/models/icnet_r50-d8.py',
    '../_base_/datasets/hots_v1_640x480.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
crop_size = (640, 480)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(size=crop_size)
load_from = "checkpoints/icnet_r50-d8_832x832_160k_cityscapes_20210925_232612-a95f0d4e.pth"
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=46
    ),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=128,
            num_convs=1,
            num_classes=46,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=128,
            num_convs=1,
            num_classes=46,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ]
    
    )
