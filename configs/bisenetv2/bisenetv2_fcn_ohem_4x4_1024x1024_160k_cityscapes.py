_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# sampler = dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=16,
            channels=16,
            num_convs=2,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=32,
            channels=64,
            num_convs=2,
            num_classes=19,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=256,
            num_convs=2,
            num_classes=19,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=1024,
            num_convs=2,
            num_classes=19,
            in_index=4,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
)

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.05)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
