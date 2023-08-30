# dataset settings
dataset_type = 'NYUDataset'
data_root = 'data/nyu'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(dict(type='LoadDepthAnnotation', depth_rescale_factor=1e-3)),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'category_id'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root, pipeline=[]))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(
            img_path='images/test', depth_map_path='annotations/test'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DepthMetric', max_depth_eval=10.0, crop_type='nyu')
test_evaluator = val_evaluator
