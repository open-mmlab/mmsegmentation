_base_ = './upernet_mae-base_fp16_8x2_512x512_160k_ade20k.py'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # TODO: Refactor 'MultiScaleFlipAug' which supports
    # `min_size` feature in `Resize` class
    # img_ratios is [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    # original image scale is (2048, 512)
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='PackSegInputs')
]
val_dataloader = dict(batch_size=2, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
