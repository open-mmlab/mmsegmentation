_base_ = [
    '../_base_/models/cgnet.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='Adam', lr=0.001, eps=1e-08, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
total_iters = 60000
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')

img_norm_cfg = dict(
    mean=[72.39239876, 82.90891754, 73.15835921], std=[1, 1, 1], to_rgb=True)
crop_size = (680, 680)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=crop_size),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=8, num_workers=4, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8, num_workers=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
