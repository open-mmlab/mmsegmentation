_base_ = [
    '../_base_/models/psp_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py'
]
crop_size = (713, 713)
cudnn_benchmark = True
# model training and testing settings
train_cfg = dict(sampler=None)
test_cfg = dict(
    mode='slide',
    crop_size=crop_size,
    stride=(476, 476),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_seg=True),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomRotate', rotate_range=(-10, 10), rotate_ratio=0.5),
    dict(type='RandomGaussianBlur', blur_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=[(1024, 512), (1536, 768), (2048, 1024),
        #            (2560, 1280), (3072, 1536), (4096, 2048)],
        img_scale=(2048, 1024),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(name_lr_mult=[dict(name='head', lr_mult=10)]))
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    by_epoch=False,
)
# runtime settings
total_epochs = 20
evaluation = dict(interval=1, metric='mIoU')
