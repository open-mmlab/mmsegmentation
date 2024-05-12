_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/hots_v1_640x480.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (640, 480)
data_preprocessor = dict(size=crop_size)
load_from = "checkpoints/fcn_d6_r50-d16_512x1024_40k_cityscapes_20210305_130133-98d5d1bc.pth"
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(
        dilation=6,
        num_classes=46
        ),
    auxiliary_head=dict(
        dilation=6,
        num_classes=46
        )
    )
