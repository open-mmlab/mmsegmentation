_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(decode_head=dict(num_classes=150))
optimizer = dict(paramwise_cfg=dict(norm_decay_mult=0))
