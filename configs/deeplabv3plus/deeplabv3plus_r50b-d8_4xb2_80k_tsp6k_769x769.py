_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/tsp6k_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (769, 769)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(align_corners=True, num_classes=21),
    auxiliary_head=dict(align_corners=True, num_classes=21),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
