_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(
        dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1), multi_grid=(1, 2, 4)),
    decode_head=dict(dilations=(1, 6, 12, 18)))
