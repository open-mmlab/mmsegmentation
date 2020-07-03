_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/cityscapes_769x769.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(decode_head=dict(align_corners=True))
test_cfg = dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
optimizer = dict(paramwise_cfg=dict(norm_decay_mult=0))
