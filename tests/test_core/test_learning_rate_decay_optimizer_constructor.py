# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.core.utils.layer_decay_optimizer_constructor import (
    LearningRateDecayOptimizerConstructor, get_num_layer_layer_wise,
    get_num_layer_stage_wise)

base_lr = 0.0001
base_wd = 0.05
momentum = 0.9
weight_decay = 0.05


class ConvNeXtExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleList()
        self.backbone.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(ConvModule(3, 4, kernel_size=1, bias=True))
            self.backbone.stages.append(stage)
        self.backbone.norm0 = nn.BatchNorm2d(2)

        # add some variables to meet unit test coverate rate
        self.backbone.cls_token = nn.Parameter(torch.ones(1))
        self.backbone.mask_token = nn.Parameter(torch.ones(1))
        self.backbone.pos_embed = nn.Parameter(torch.ones(1))
        self.backbone.stem_norm = nn.Parameter(torch.ones(1))
        self.backbone.downsample_norm0 = nn.BatchNorm2d(2)
        self.backbone.downsample_norm1 = nn.BatchNorm2d(2)
        self.backbone.downsample_norm2 = nn.BatchNorm2d(2)
        self.backbone.lin = nn.Parameter(torch.ones(1))
        self.backbone.lin.requires_grad = False
        self.backbone.downsample_layers = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1, bias=True))

        self.decode_head = nn.Conv2d(2, 2, kernel_size=1, groups=2)


class PseudoDataParallel(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = ConvNeXtExampleModel()

    def forward(self, x):
        return x


def check_convnext_adamw_optimizer(optimizer, paramwise_cfg):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd
    param_groups = optimizer.param_groups
    assert len(param_groups) == 12
    for param_dict in param_groups:

        # Just pick up the first variable in each groups
        example_param_name = param_dict['param_names'][0]
        example_param = param_dict['params'][0]

        num_layers = paramwise_cfg['num_layers'] + 2
        if len(example_param.shape) == 1 or example_param_name.endswith(
                '.bias') or example_param_name in ('pos_embed', 'cls_token'):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay

        if paramwise_cfg['decay_type'] == 'layer_wise':
            layer_id = get_num_layer_layer_wise(example_param_name,
                                                paramwise_cfg['num_layers'])
        elif paramwise_cfg['decay_type'] == 'stage_wise':
            layer_id = get_num_layer_stage_wise(example_param_name, num_layers)
        group_name = f'layer_{layer_id}_{group_name}'

        scale = paramwise_cfg['decay_rate']**(num_layers - layer_id - 1)
        assert this_weight_decay == param_dict['weight_decay']
        assert scale == param_dict['lr_scale']
        assert scale * optimizer.defaults['lr'] == param_dict['lr']
        assert group_name == param_dict['group_name']


def test_convnext_learning_rate_decay_optimizer_constructor():

    # paramwise_cfg with ConvNeXtExampleModel
    model = ConvNeXtExampleModel()
    optimizer_cfg = dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
    stagewise_paramwise_cfg = dict(
        decay_rate=0.9, decay_type='stage_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg, stagewise_paramwise_cfg)
    optimizer = optim_constructor(model)
    check_convnext_adamw_optimizer(optimizer, stagewise_paramwise_cfg)

    layerwise_paramwise_cfg = dict(
        decay_rate=0.9, decay_type='layer_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg, layerwise_paramwise_cfg)
    optimizer = optim_constructor(model)
    check_convnext_adamw_optimizer(optimizer, layerwise_paramwise_cfg)
