_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth'  # noqa

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        img_size=(512, 512),
        embed_dims=384,
        num_heads=6,
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=384,
        channels=384,
        num_classes=150,
        num_layers=2,
        num_heads=6,
        embed_dims=384,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

optimizer = dict(lr=0.001, weight_decay=0.0)

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=crop_size),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=1,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
