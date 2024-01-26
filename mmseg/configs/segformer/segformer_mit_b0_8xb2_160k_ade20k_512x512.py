# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .._base_.models.segformer_mit_b0 import *  # noqa: F401,F403
    from .._base_.datasets.ade20k import *  # noqa: F401,F403
    from .._base_.schedules.schedule_160k import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.model.weight_init import PretrainedInit
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.optim.adamw import AdamW

crop_size = (512, 512)
data_preprocessor.update(size=crop_size)  # noqa: F405
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model.update(  # noqa: F405
    data_preprocessor=data_preprocessor,  # noqa: F405
    backbone=dict(init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint)),
    decode_head=dict(num_classes=150))

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
train_dataloader.update(batch_size=2, num_workers=2)  # noqa: F405
val_dataloader.update(batch_size=1, num_workers=4)  # noqa: F405
test_dataloader = val_dataloader  # noqa: F405
