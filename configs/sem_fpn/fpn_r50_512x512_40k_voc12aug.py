_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(decode_head=dict(num_classes=21))
optimizer = dict(paramwise_cfg=dict(norm_decay_mult=0))
