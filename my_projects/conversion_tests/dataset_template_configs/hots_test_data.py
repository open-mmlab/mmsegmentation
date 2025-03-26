data_root = '/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/'
dataset_type = 'HOTSDataset'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        data_root='/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='HOTSDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type="CustomIoUMetricConversion")
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
