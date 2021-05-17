_base_ = [
    '../_base_/models/setr_naive.py', '../_base_/datasets/pascal_context.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(img_size=(480, 480), drop_rate=0.),
    decode_head=dict(num_classes=60),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=0,
            embed_dim=1024,
            num_classes=60,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=1,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=1,
            embed_dim=1024,
            num_classes=60,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=1,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=512,
            in_index=2,
            embed_dim=1024,
            num_classes=60,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=1,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))

optimizer = dict(
    lr=0.01,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}),
)

data = dict(samples_per_gpu=2)
