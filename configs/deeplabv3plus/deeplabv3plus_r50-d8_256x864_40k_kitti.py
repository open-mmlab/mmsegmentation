_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/kitti_seg.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

model = dict(
    test_cfg=dict(mode='slide', crop_size=(368, 368), stride=(246, 246)))
