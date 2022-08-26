_base_ = [
    '../_base_/models/cgnet.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='Adam', lr=0.001, eps=1e-08, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        by_epoch=False,
        begin=0,
        end=60000)
]
# runtime settings
total_iters = 60000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=total_iters, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'))

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
