# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['../../../configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py']

custom_imports = dict(imports=['projects.example_project.dummy'])

model = dict(backbone=dict(type='DummyResNet'))
