_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(
        num_classes=21, type='FCNDilateHead', dilation=6, concat_input=True),
    auxiliary_head=dict(
        num_classes=21, type='FCNDilateHead', dilation=6, concat_input=True))
