data_root = '/media/ids/Ubuntu files/data/irl_vision_sim_cat/SemanticSegmentation/'
dataset_type = 'IRLVisionSimCATDataset'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        data_root=
        '/media/ids/Ubuntu files/data/irl_vision_sim_cat/SemanticSegmentation/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='IRLVisionSimCATDataset'),
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
