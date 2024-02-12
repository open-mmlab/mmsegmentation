_base_ = [
    '../_base_/models/ham_mscan.py',
    '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=crop_size))
decode_head=dict(num_classes=6)

