_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(img_size=(512, 512), drop_rate=0.),
    decode_head=dict(num_classes=150),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=0,
            embed_dim=1024,
            num_classes=150,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=1,
            embed_dim=1024,
            num_classes=150,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=2,
            embed_dim=1024,
            num_classes=150,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=3,
            embed_dim=1024,
            num_classes=150,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

optimizer = dict(
    lr=0.001,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

data = dict(samples_per_gpu=2)
