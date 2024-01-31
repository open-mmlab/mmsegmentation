_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/beachtypes_c.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_25k.py'
]

crop_size = (500, 1500)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=crop_size))

