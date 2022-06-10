_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 1024)
preprocess_cfg = dict(size=crop_size)
model = dict(preprocess_cfg=preprocess_cfg)

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
