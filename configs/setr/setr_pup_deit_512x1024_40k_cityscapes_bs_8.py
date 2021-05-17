_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained=\
    'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',  # noqa
    backbone=dict(
        drop_rate=0.,
        img_size=(512, 1024),
        out_indices=(4, 7, 9, 11),
        embed_dim=768,
        depth=12,
        num_heads=12),
    decode_head=dict(
        in_channels=768,
        embed_dim=768,
    ),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=512,
            in_index=0,
            embed_dim=768,
            num_classes=19,
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
            in_channels=768,
            channels=512,
            in_index=1,
            embed_dim=768,
            num_classes=19,
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
            in_channels=768,
            channels=512,
            in_index=2,
            embed_dim=768,
            num_classes=19,
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
            in_channels=768,
            channels=512,
            in_index=3,
            embed_dim=768,
            num_classes=19,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_mode='bilinear',
            num_up_layer=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
)

optimizer = dict(
    lr=0.01,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}),
)

data = dict(samples_per_gpu=1)
