_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    dict(type='PackSegInputs')
]
test_dataloader = dict(
    dataset=dict(
        type='NightDrivingDataset',
        data_root='data/NighttimeDrivingTest/',
        data_prefix=dict(
            img_path='leftImg8bit/test/night',
            seg_map_path='gtCoarse_daytime_trainvaltest/test/night'),
        pipeline=test_pipeline))
