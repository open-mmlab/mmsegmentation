_base_ = [
    '../_base_/models/ocrnetplus_r50-d8_sep.py', '../_base_/datasets/cityscapes_bs2x.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_lr2x.py'
]
