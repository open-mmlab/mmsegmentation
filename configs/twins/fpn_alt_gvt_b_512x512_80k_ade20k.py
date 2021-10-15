_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/alt_gvt_base.pth',
    backbone=dict(
        type='alt_gvt_base',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150),
    )

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU')

data = dict(
    samples_per_gpu=4)