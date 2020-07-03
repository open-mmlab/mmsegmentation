_base_ = [
    '../_base_/models/pointrend_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
optimizer = dict(paramwise_cfg=dict(norm_decay_mult=0))
