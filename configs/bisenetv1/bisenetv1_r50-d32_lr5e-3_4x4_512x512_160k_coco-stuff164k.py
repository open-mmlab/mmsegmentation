_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/coco-stuff164k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_channels=1024,
        backbone_cfg=dict(type='ResNet', depth=50)),
    decode_head=dict(in_channels=1024, channels=1024, num_classes=171),
    auxiliary_head=[
        dict(in_channels=512, channels=256, num_classes=171),
        dict(in_channels=512, channels=256, num_classes=171),
    ])
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.005)
