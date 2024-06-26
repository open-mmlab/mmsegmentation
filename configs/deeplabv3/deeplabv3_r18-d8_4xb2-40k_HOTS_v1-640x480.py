_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/hots_v1_640x480.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (640, 480)
data_preprocessor = dict(size=crop_size)
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=46
    ),
    auxiliary_head=dict(
        in_channels=256, 
        channels=64,
        num_classes=46),
    data_preprocessor=data_preprocessor)
