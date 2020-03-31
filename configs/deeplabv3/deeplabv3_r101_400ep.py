_base_ = [
    '../_base_/models/deeplabv3_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py'
]
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
crop_size = (769, 769)
cudnn_benchmark = True
# model training and testing settings
train_cfg = dict(sampler=None)
test_cfg = dict(
    mode='slide',
    crop_size=crop_size,
    stride=(513, 513),
)
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
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(dataset=dict(pipeline=train_pipeline)))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    by_epoch=False,
)
# runtime settings
total_epochs = 40
evaluation = dict(interval=1, metric='mIoU')
