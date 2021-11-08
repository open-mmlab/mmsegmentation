_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/coco-stuff164k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetV1',
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_channels=1024,
        backbone_cfg=dict(type='ResNet', depth=50)),
    decode_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=0,
        channels=1024,
        num_classes=171),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=171,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False),
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=171,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False),
    ])
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
