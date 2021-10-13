_base_ = [
    '_base_/models/upernet_r50.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_160k.py'
]
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/pcpvt_large.pth',
    backbone=dict(
        type='pcpvt_large',
        drop_path_rate=0.3,
        style='pytorch'),
    decode_head=dict(num_classes=150, in_channels=[64, 128, 320, 512]),
    auxiliary_head=dict(num_classes=150, in_channels=320)
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=2)