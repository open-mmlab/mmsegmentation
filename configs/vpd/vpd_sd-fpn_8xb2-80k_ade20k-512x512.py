_base_ = [
    '../_base_/models/vpd_sd.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

find_unused_parameters = True

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
)

model = dict(
    backbone=dict(class_embed_path='vpd/seg_class_embeddings.pth'),
    neck=dict(
        type='FPN',
        in_channels=[320, 790, 1430, 1280],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.00008, weight_decay=0.001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.unet': dict(lr_mult=0.1),
            'backbone.encoder_vq': dict(lr_mult=0.0),
            'backbone.text_encoder': dict(lr_mult=0.0)
        }),
    clip_grad=None)

# optimizer = dict(_delete_=True, type='AdamW', lr=0.00008, weight_decay=0.001)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None,
# custom_keys={'backbone.unet': dict(lr_mult=0.1),
#              'backbone.encoder_vq': dict(lr_mult=0.0),
#              'backbone.text_encoder': dict(lr_mult=0.0)},
# )

# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=1, end=1500, by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False)
]
