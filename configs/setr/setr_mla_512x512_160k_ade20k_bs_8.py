_base_ = [
    '../_base_/models/setr_mla.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
aux_alpha = dict(
    type='SETRMLAAUXHead',
    in_channels=(1024, 1024, 1024, 1024),
    channels=512,
    in_index=(0, 1, 2, 3),
    img_size=(512, 512),
    embed_dim=1024,
    mla_channels=256,
    mla_select_index=0,
    num_classes=150,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_beta = dict(
    type='SETRMLAAUXHead',
    in_channels=(1024, 1024, 1024, 1024),
    channels=512,
    in_index=(0, 1, 2, 3),
    img_size=(512, 512),
    embed_dim=1024,
    mla_channels=256,
    mla_select_index=1,
    num_classes=150,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_gamma = dict(
    type='SETRMLAAUXHead',
    in_channels=(1024, 1024, 1024, 1024),
    channels=512,
    in_index=(0, 1, 2, 3),
    img_size=(512, 512),
    embed_dim=1024,
    mla_channels=256,
    mla_select_index=2,
    num_classes=150,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_delta = dict(
    type='SETRMLAAUXHead',
    in_channels=(1024, 1024, 1024, 1024),
    channels=512,
    in_index=(0, 1, 2, 3),
    img_size=(512, 512),
    embed_dim=1024,
    mla_channels=256,
    mla_select_index=3,
    num_classes=150,
    norm_cfg=norm_cfg,
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

find_unused_parameters = True
data = dict(samples_per_gpu=1)
