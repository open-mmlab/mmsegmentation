_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
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
