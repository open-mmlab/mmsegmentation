_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(
        dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1), multi_grid=(1, 2, 4)),
    decode_head=dict(
        dilations=(1, 6, 12, 18),
        sampler=dict(type='OHEMPixelSampler', min_kept=100000)))
lr_config = dict(warmup='linear', warmup_ratio=1 / 200, warmup_iters=200)
