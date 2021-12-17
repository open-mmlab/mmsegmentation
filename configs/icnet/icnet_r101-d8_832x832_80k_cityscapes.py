# Copyright (c) OpenMMLab. All rights reserved.
_base_ = './icnet_r50-d8_832x832_80k_cityscapes.py'
model = dict(backbone=dict(backbone_cfg=dict(depth=101)))
