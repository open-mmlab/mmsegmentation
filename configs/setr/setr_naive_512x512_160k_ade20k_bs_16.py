_base_ = [
    '../_base_/models/setr_naive.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
aux_alpha = dict(
    type='SETRUPHead',
    in_channels=1024,
    channels=512,
    in_index=0,
    img_size=(512, 512),
    embed_dim=1024,
    num_classes=150,
    norm_cfg=norm_cfg,
    num_convs=2,
    up_mode='bilinear',
    num_up_layer=1,
    conv3x3_conv1x1=False,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_beta = dict(
    type='SETRUPHead',
    in_channels=1024,
    channels=512,
    in_index=1,
    img_size=(512, 512),
    embed_dim=1024,
    num_classes=150,
    norm_cfg=norm_cfg,
    num_convs=2,
    up_mode='bilinear',
    num_up_layer=1,
    conv3x3_conv1x1=False,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
aux_gamma = dict(
    type='SETRUPHead',
    in_channels=1024,
    channels=512,
    in_index=2,
    img_size=(512, 512),
    embed_dim=1024,
    num_classes=150,
    norm_cfg=norm_cfg,
    num_convs=2,
    up_mode='bilinear',
    num_up_layer=1,
    conv3x3_conv1x1=False,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
model = dict(
    backbone=dict(img_size=(512, 512), drop_rate=0.),
    decode_head=dict(img_size=(512, 512), num_classes=150),
    auxiliary_head=[aux_alpha, aux_beta, aux_gamma],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

optimizer = dict(
    lr=0.01,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

find_unused_parameters = False
data = dict(samples_per_gpu=2)
