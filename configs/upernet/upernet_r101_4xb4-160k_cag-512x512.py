_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/cag.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150),
    pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
