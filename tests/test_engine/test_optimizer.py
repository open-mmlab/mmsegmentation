# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.optim import build_optim_wrapper


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x):
        return x


base_lr = 0.01
base_wd = 0.0001
momentum = 0.9


def test_build_optimizer():
    model = ExampleModel()
    optim_wrapper_cfg = dict(
        type='OptimWrapper',
        optimizer=dict(
            type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum))
    optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
    # test whether optimizer is successfully built from parent.
    assert isinstance(optim_wrapper.optimizer, torch.optim.SGD)
