_base_ = [
    '../../../configs/_base_/models/pspnet_r50-d8.py',
    './_base_/datasets/mapillary_v2.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_240k.py'
]
custom_imports = dict(
    imports=['projects.mapillary_dataset.mmseg.datasets.mapillary'])
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(num_classes=124),
    auxiliary_head=dict(num_classes=124))
