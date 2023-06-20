_base_ = [
    '../../../configs/_base_/models/deeplabv3plus_r50-d8.py',
    './_base_/datasets/gid.py', '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_240k.py'
]
custom_imports = dict(imports=['projects.gid_dataset.mmseg.datasets.gid'])

crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6))
