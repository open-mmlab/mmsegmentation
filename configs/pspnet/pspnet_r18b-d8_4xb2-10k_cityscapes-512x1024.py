_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/hots_v1_640x480.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_10k.py'
]

crop_size = (640, 480)
data_preprocessor = dict(size=crop_size)


model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
