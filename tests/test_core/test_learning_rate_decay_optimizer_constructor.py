# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmseg.core.utils.layer_decay_optimizer_constructor import \
    LearningRateDecayOptimizerConstructor


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


base_lr = 0.01
base_wd = 0.0001
momentum = 0.9


class PseudoDataParallel(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = ExampleModel()

    def forward(self, x):
        return x


def check_default_optimizer(optimizer, model, prefix=''):
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    param_groups = optimizer.param_groups[0]

    param_names = [
        'param1', 'conv1.weight', 'conv2.weight', 'conv2.bias', 'bn.weight',
        'bn.bias', 'sub.param1', 'sub.conv1.weight', 'sub.conv1.bias',
        'sub.gn.weight', 'sub.gn.bias'
    ]
    param_dict = dict(model.named_parameters())
    assert len(param_groups['params']) == len(param_names)
    for i in range(len(param_groups['params'])):
        assert torch.equal(param_groups['params'][i],
                           param_dict[prefix + param_names[i]])


def test_learning_rate_decay_optimizer_constructor():
    model = ExampleModel()

    with pytest.raises(TypeError):
        # optimizer_cfg must be a dict
        optimizer_cfg = []
        optim_constructor = LearningRateDecayOptimizerConstructor(
            optimizer_cfg)
        optim_constructor(model)

    with pytest.raises(TypeError):
        # paramwise_cfg must be a dict or None
        optimizer_cfg = dict(lr=0.0001)
        paramwise_cfg = ['error']
        optim_constructor = LearningRateDecayOptimizerConstructor(
            optimizer_cfg, paramwise_cfg)
        optim_constructor(model)

    with pytest.raises(ValueError):
        # bias_decay_mult/norm_decay_mult is specified but weight_decay is None
        optimizer_cfg = dict(lr=0.0001, weight_decay=None)
        paramwise_cfg = dict(bias_decay_mult=1, norm_decay_mult=1)
        optim_constructor = LearningRateDecayOptimizerConstructor(
            optimizer_cfg, paramwise_cfg)
        optim_constructor(model)

    # basic config with ExampleModel
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    optim_constructor = LearningRateDecayOptimizerConstructor(optimizer_cfg)
    optimizer = optim_constructor(model)
    check_default_optimizer(optimizer, model)

    # basic config with pseudo data parallel
    model = PseudoDataParallel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = None
    optim_constructor = LearningRateDecayOptimizerConstructor(optimizer_cfg)
    optimizer = optim_constructor(model)
    check_default_optimizer(optimizer, model, prefix='module.')

    # basic config with DataParallel
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(ExampleModel())
        optimizer_cfg = dict(
            type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
        paramwise_cfg = None
        optim_constructor = LearningRateDecayOptimizerConstructor(
            optimizer_cfg)
        optimizer = optim_constructor(model)
        check_default_optimizer(optimizer, model, prefix='module.')

    # Empty paramwise_cfg with ExampleModel
    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = dict()
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg, paramwise_cfg)
    optimizer = optim_constructor(model)
    check_default_optimizer(optimizer, model)

    # Empty paramwise_cfg with ExampleModel and no grad
    model = ExampleModel()
    for param in model.parameters():
        param.requires_grad = False
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = dict()
    optim_constructor = LearningRateDecayOptimizerConstructor(optimizer_cfg)
    optimizer = optim_constructor(model)
    check_default_optimizer(optimizer, model)

    # WIP TODO: ADD UNIT TEST

    # # paramwise_cfg with ExampleModel
    # model = ExampleModel()
    # optimizer_cfg = dict(
    #     type='AdamW', lr=base_lr, weight_decay=base_wd, momentum=momentum,
    #     betas=(0.9, 0.999),
    # )
    # paramwise_cfg={
    #     'decay_rate': 0.9,
    #     'decay_type': 'stage_wise',
    #     'num_layers': 12
    # }
    # optim_constructor = LearningRateDecayOptimizerConstructor(optimizer_cfg,
    #                                                 paramwise_cfg)
    # optimizer = optim_constructor(model)
    # check_default_optimizer(optimizer, model)
