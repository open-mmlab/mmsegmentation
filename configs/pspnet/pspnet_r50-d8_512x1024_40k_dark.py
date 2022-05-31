_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
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
        type='DarkZurichDataset',
        data_root='data/dark_zurich/',
        data_prefix=dict(
            img_path='rgb_anon/val/night/GOPR0356',
            seg_map_path='gt/val/night/GOPR0356'),
        pipeline=test_pipeline))
