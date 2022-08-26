_base_ = './beit-large_upernet_8xb1-amp-160k_ade20k-640x640.py'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # TODO: Refactor 'MultiScaleFlipAug' which supports
    # `min_size` feature in `Resize` class
    # img_ratios is [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    # original image scale is (2560, 640)
    dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs'),
]
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
