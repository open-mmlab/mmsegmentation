_base_ = [
    '../_base_/models/ocrnet_r50-d8-align.py', '../_base_/datasets/cityscapes_769x769.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

test_cfg = dict(mode='slide', crop_size=(769, 769), stride=(513, 513))