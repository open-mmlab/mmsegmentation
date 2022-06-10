_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
preprocess_cfg = dict(size=crop_size)
model = dict(preprocess_cfg=preprocess_cfg, decode_head=dict(num_classes=21))
