_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/isaid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (896, 896)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=16),
    auxiliary_head=dict(num_classes=16))
