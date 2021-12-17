# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
