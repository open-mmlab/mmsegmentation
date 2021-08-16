_base_ = [
    '../_base_/models/setr_mla.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(img_size=(512, 512), drop_rate=0.),
    decode_head=dict(num_classes=150),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=0,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=150,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=1,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=150,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=2,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=150,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=3,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=150,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

optimizer = dict(
    lr=0.001,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

# num_gpus: 8 -> batch_size: 8
data = dict(samples_per_gpu=1)
