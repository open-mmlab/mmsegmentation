_base_ = [
    '../_base_/models/pointrend_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=200),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=200,
        end=80000,
        by_epoch=False,
    )
]
