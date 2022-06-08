# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# training schedule for 320k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=320000, val_interval=32000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
evaluation = dict(interval=32000, metric='mIoU')
default_hooks = dict(
    optimizer=dict(type='OptimizerHook', grad_clip=None),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=32000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
