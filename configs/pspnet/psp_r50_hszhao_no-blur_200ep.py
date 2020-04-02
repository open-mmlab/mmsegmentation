_base_ = './psp_r50_tv_200ep.py'
model = dict(
    pretrained='pretrain_model/resnet50c128_hszhao-b3e6e229.pth',
    backbone=dict(deep_stem=True, base_channels=128))

crop_size = (713, 713)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_seg=True),
    dict(
        type='Resize',
        img_scale=[(1024, 512), (4096, 2048)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomRotate', rotate_range=(-10, 10), rotate_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
