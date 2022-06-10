_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/isaid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (896, 896)
preprocess_cfg = dict(size=crop_size)
model = dict(preprocess_cfg=preprocess_cfg, decode_head=dict(num_classes=16))
