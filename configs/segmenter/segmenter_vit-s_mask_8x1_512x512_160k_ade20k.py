_base_ = [
    '../_base_/models/segmenter_vit.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    backbone=dict(img_size=(512, 512)),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=384,
        channels=384,
        num_classes=150,
        num_layers=2,
        num_heads=6,
        embed_dims=384,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
)

optimizer = dict(lr=0.001, weight_decay=0.0)

# num_gpus: 8 -> batch_size: 8
data = dict(samples_per_gpu=1)

# TODO: handle img_norm_cfg
