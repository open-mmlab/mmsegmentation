# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import DefaultOptimizerConstructor

from mmseg.core.builder import (OPTIMIZER_BUILDERS, build_optimizer,
                                build_optimizer_constructor)


class SubModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, groups=2)
        self.gn = nn.GroupNorm(2, 2)
        self.param1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()

    def forward(self, x):
        return x


class ExampleDuplicateModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(4, 2, kernel_size=1))
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()
        self.conv3 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False))
        self.conv3[0] = self.conv1[0]

    def forward(self, x):
        return x


class PseudoDataParallel(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = ExampleModel()

    def forward(self, x):
        return x


base_lr = 0.01
base_wd = 0.0001
momentum = 0.9


def test_build_optimizer_constructor():
    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = dict()
    optim_constructor_cfg = dict(
        type='DefaultOptimizerConstructor',
        optimizer_cfg=optimizer_cfg,
        paramwise_cfg=paramwise_cfg)
    optim_constructor = build_optimizer_constructor(optim_constructor_cfg)
    optimizer = optim_constructor(model)
    # Test whether optimizer constructor can be built from parent.
    assert isinstance(optimizer, torch.optim.SGD)

    from mmcv.runner import OPTIMIZERS
    from mmcv.utils import build_from_cfg

    @OPTIMIZER_BUILDERS.register_module()
    class MyOptimizerConstructor(DefaultOptimizerConstructor):

        def __call__(self, model):
            if hasattr(model, 'module'):
                model = model.module

            conv1_lr_mult = self.paramwise_cfg.get('conv1_lr_mult', 1.)

            params = []
            for name, param in model.named_parameters():
                param_group = {'params': [param]}
                if name.startswith('conv1') and param.requires_grad:
                    param_group['lr'] = self.base_lr * conv1_lr_mult
                params.append(param_group)
            optimizer_cfg['params'] = params

            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

    paramwise_cfg = dict(conv1_lr_mult=5)
    optim_constructor_cfg = dict(
        type='MyOptimizerConstructor',
        optimizer_cfg=optimizer_cfg,
        paramwise_cfg=paramwise_cfg)
    optim_constructor = build_optimizer_constructor(optim_constructor_cfg)
    optimizer = optim_constructor(model)
    # Tests whether a new optimizer constructor can be registered.
    assert isinstance(optimizer, torch.optim.SGD)


def test_build_optimizer():
    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    optimizer = build_optimizer(model, optimizer_cfg)
    # test whether optimizer is successfully built from parent.
    assert isinstance(optimizer, torch.optim.SGD)
