_base_ = [
    '../_base_/models/ocrnet_sep_aspp_r50-d8.py', '../_base_/datasets/cityscapes_bs2x.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_lr2x.py'
]
