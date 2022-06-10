_base_ = [
    '../_base_/models/psanet_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
preprocess_cfg = dict(size=crop_size)
model = dict(
    preprocess_cfg=preprocess_cfg,
    decode_head=dict(mask_size=(66, 66), num_classes=150),
    auxiliary_head=dict(num_classes=150))
