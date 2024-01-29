# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import PolyLR
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop
# from mmengine.runner.loops import EpochBasedTrainLoop
from torch.optim.sgd import SGD

from mmseg.engine import SegVisualizationHook

# optimizer
optimizer = dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type=OptimWrapper, optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=320000,
        by_epoch=False)
]
# training schedule for 320k
train_cfg = dict(type=IterBasedTrainLoop, max_iters=320000, val_interval=32000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=32000),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))
