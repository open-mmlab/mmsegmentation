# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import ConstantLR, LinearLR
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop
# from mmengine.runner.loops import EpochBasedTrainLoop
from torch.optim.adamw import AdamW

from mmseg.engine import SegVisualizationHook
from mmseg.engine.schedulers import PolyLRRatio

# optimizer
optimizer = dict(type=AdamW, lr=0.01, weight_decay=0.1)

optim_wrapper = dict(type=OptimWrapper, optimizer=optimizer, clip_grad=None)
# learning policy

# learning policy
param_scheduler = [
    dict(type=LinearLR, start_factor=3e-2, begin=0, end=12000, by_epoch=False),
    dict(
        type=PolyLRRatio,
        eta_min_ratio=3e-2,
        power=0.9,
        begin=12000,
        end=24000,
        by_epoch=False),
    dict(type=ConstantLR, by_epoch=False, factor=1, begin=24000, end=25000)
]

# training schedule for 25k
train_cfg = dict(type=IterBasedTrainLoop, max_iters=25000, val_interval=1000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=1000),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))
