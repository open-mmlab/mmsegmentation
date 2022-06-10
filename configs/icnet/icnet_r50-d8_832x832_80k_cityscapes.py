_base_ = [
    '../_base_/models/icnet_r50-d8.py',
    '../_base_/datasets/cityscapes_832x832.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (832, 832)
preprocess_cfg = dict(size=crop_size)
model = dict(preprocess_cfg=preprocess_cfg)
