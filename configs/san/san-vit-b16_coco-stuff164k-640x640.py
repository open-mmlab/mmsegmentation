_base_ = [
    '../_base_/models/san_vit-b16.py', '../_base_/datasets/coco-stuff164k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(640 * x * 0.1) for x in range(5, 16)],
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

model = dict(
    pretrained='pretrain/clip_vit_base_patch16_224.pth',
    text_encoder=dict(dataset_name='coco-stuff164k'),
    decode_head=dict(num_classes=171))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=160000,
        by_epoch=False,
    )
]
