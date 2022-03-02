# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcls.models.backbones import ConvNeXt

from mmseg.core.utils.layer_decay_optimizer_constructor import (
    LearningRateDecayOptimizerConstructor, get_num_layer_layer_wise,
    get_num_layer_stage_wise)
from mmseg.models.decode_heads import UPerHead

base_lr = 0.0001
base_wd = 0.05
momentum = 0.9
weight_decay = 0.05


class ConvNeXtExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXt(
            arch='base',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False)

        self.decode_head = UPerHead(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=19)

    def forward(self, x):
        return x


class PseudoDataParallel(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = ConvNeXtExampleModel()

    def forward(self, x):
        return x


def check_convnext_adamw_optimizer(optimizer,
                                   model,
                                   decay_rate=0.9,
                                   decay_type='stage_wise',
                                   num_layers=6):
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd

    # Get group number of each parameters in ConvNeXt model,
    # which is lost by its `.values()` operations in the last.
    group_name_lst = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or name in (
                'pos_embed', 'cls_token'):
            group_name = 'no_decay'
        else:
            group_name = 'decay'
        if decay_type == 'layer_wise':
            layer_id = get_num_layer_layer_wise(
                name,
                LearningRateDecayOptimizerConstructor.paramwise_cfg.get(
                    'num_layers'))
        elif decay_type == 'stage_wise':
            layer_id = get_num_layer_stage_wise(name, num_layers)
        group_name = f'layer_{layer_id}_{group_name}'
        group_name_lst.append(group_name)

    assert len(param_groups) == len(set(group_name_lst))


def test_convnext_learning_rate_decay_optimizer_constructor():

    # paramwise_cfg with ConvNeXtExampleModel
    model = ConvNeXtExampleModel()
    optimizer_cfg = dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
    paramwise_cfg = dict(decay_rate=0.9, decay_type='stage_wise', num_layers=6)

    optim_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg, paramwise_cfg)
    optimizer = optim_constructor(model)
    check_convnext_adamw_optimizer(optimizer, model, **paramwise_cfg)
