_base_ = [
    '../_base_/models/setr_mla.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
aux_alpha = dict(
    type='SETRMLAAUXHead',
    in_channels=256,
    channels=512,
    in_index=0,
    img_size=(512, 512),
    mla_channels=256,
    num_classes=150,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_beta = dict(
    type='SETRMLAAUXHead',
    in_channels=256,
    channels=512,
    in_index=1,
    img_size=(512, 512),
    mla_channels=256,
    num_classes=150,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_gamma = dict(
    type='SETRMLAAUXHead',
    in_channels=256,
    channels=512,
    in_index=2,
    img_size=(512, 512),
    mla_channels=256,
    num_classes=150,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_delta = dict(
    type='SETRMLAAUXHead',
    in_channels=256,
    channels=512,
    in_index=3,
    img_size=(512, 512),
    mla_channels=256,
    num_classes=150,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
model = dict(
    backbone=dict(img_size=(512, 512), drop_rate=0.),
    decode_head=dict(img_size=(512, 512), num_classes=150),
    auxiliary_head=[aux_alpha, aux_beta, aux_gamma, aux_delta],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

optimizer = dict(
    lr=0.001,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

data = dict(samples_per_gpu=1)
