# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    '../_base_/models/erfnet_fcn.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
