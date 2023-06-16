_base_ = [
    '../../../../configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/kittistep.py',
    '../../../../configs/_base_/default_runtime.py',
    '../../../../configs/_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
