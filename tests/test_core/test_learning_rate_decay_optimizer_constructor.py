# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcls.models.backbones import ConvNeXt

from mmseg.core.utils.layer_decay_optimizer_constructor import \
    LearningRateDecayOptimizerConstructor
from mmseg.models.decode_heads import UPerHead

base_lr = 0.0001
base_wd = 0.05
momentum = 0.9
weight_decay = 0.05


class ConvNeXtExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXt(
            arch='tiny',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False)
        self.backbone.cls_token = nn.Parameter(torch.ones(1))
        self.backbone.mask_token = nn.Parameter(torch.ones(1))
        self.backbone.pos_embed = nn.Parameter(torch.ones(1))
        self.backbone.stem_norm = nn.Parameter(torch.ones(1))
        self.backbone.downsample_norm0 = nn.BatchNorm2d(2)
        self.backbone.downsample_norm1 = nn.BatchNorm2d(2)
        self.backbone.downsample_norm2 = nn.BatchNorm2d(2)
        self.backbone.lin = nn.Parameter(torch.ones(1))
        self.backbone.lin.requires_grad = False

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


def check_convnext_adamw_optimizer(optimizer):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd


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
    check_convnext_adamw_optimizer(optimizer)

    layerwise_paramwise_cfg = dict(
        decay_rate=0.9, decay_type='layer_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg, layerwise_paramwise_cfg)
    optimizer = optim_constructor(model)
    check_convnext_adamw_optimizer(optimizer)
