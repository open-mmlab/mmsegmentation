# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0002)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=400),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=400,
        end=4000,
        by_epoch=False,
    )
]
# training schedule for 20k
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=4000,
    val_interval=100,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='TestBestModelCheckpointHook',
        by_epoch=False,
        save_best="val/mDice",
        rule="greater",
        load_best_for_testing='val/mDice',
        save_last=False,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=1,
        modes=["test"],
    ),
)
custom_hooks = [
    dict(type='ForceRunTestLoop'),
]