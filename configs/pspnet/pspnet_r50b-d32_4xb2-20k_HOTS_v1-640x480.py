_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/hots_v1_640x480.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

crop_size = (640, 480)
data_preprocessor = dict(size=crop_size)
num_classes = 46
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet', 
        dilations=(1, 1, 2, 4), 
        strides=(1, 2, 2, 2)
    ),
    decode_head=dict(
        num_classes=num_classes
    ),
    auxiliary_head=dict(
        num_classes=num_classes
    )
)
