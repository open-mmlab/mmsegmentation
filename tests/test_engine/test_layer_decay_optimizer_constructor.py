# Copyright (c) OpenMMLab. All rights reserved.
# from copyreg import constructor
import pytest
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.optim.optimizer import build_optim_wrapper

from mmseg.engine.optimizers.layer_decay_optimizer_constructor import \
    LearningRateDecayOptimizerConstructor
from mmseg.utils import register_all_modules

register_all_modules()

base_lr = 1
decay_rate = 2
base_wd = 0.05
weight_decay = 0.05

expected_stage_wise_lr_wd_convnext = [{
    'weight_decay': 0.0,
    'lr_scale': 128
}, {
    'weight_decay': 0.0,
    'lr_scale': 1
}, {
    'weight_decay': 0.05,
    'lr_scale': 64
}, {
    'weight_decay': 0.0,
    'lr_scale': 64
}, {
    'weight_decay': 0.05,
    'lr_scale': 32
}, {
    'weight_decay': 0.0,
    'lr_scale': 32
}, {
    'weight_decay': 0.05,
    'lr_scale': 16
}, {
    'weight_decay': 0.0,
    'lr_scale': 16
}, {
    'weight_decay': 0.05,
    'lr_scale': 8
}, {
    'weight_decay': 0.0,
    'lr_scale': 8
}, {
    'weight_decay': 0.05,
    'lr_scale': 128
}, {
    'weight_decay': 0.05,
    'lr_scale': 1
}]

expected_layer_wise_lr_wd_convnext = [{
    'weight_decay': 0.0,
    'lr_scale': 128
}, {
    'weight_decay': 0.0,
    'lr_scale': 1
}, {
    'weight_decay': 0.05,
    'lr_scale': 64
}, {
    'weight_decay': 0.0,
    'lr_scale': 64
}, {
    'weight_decay': 0.05,
    'lr_scale': 32
}, {
    'weight_decay': 0.0,
    'lr_scale': 32
}, {
    'weight_decay': 0.05,
    'lr_scale': 16
}, {
    'weight_decay': 0.0,
    'lr_scale': 16
}, {
    'weight_decay': 0.05,
    'lr_scale': 2
}, {
    'weight_decay': 0.0,
    'lr_scale': 2
}, {
    'weight_decay': 0.05,
    'lr_scale': 128
}, {
    'weight_decay': 0.05,
    'lr_scale': 1
}]

expected_layer_wise_wd_lr_beit = [{
    'weight_decay': 0.0,
    'lr_scale': 16
}, {
    'weight_decay': 0.05,
    'lr_scale': 8
}, {
    'weight_decay': 0.0,
    'lr_scale': 8
}, {
    'weight_decay': 0.05,
    'lr_scale': 4
}, {
    'weight_decay': 0.0,
    'lr_scale': 4
}, {
    'weight_decay': 0.05,
    'lr_scale': 2
}, {
    'weight_decay': 0.0,
    'lr_scale': 2
}, {
    'weight_decay': 0.05,
    'lr_scale': 1
}, {
    'weight_decay': 0.0,
    'lr_scale': 1
}]


class ToyConvNeXt(nn.Module):

    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(ConvModule(3, 4, kernel_size=1, bias=True))
            self.stages.append(stage)
        self.norm0 = nn.BatchNorm2d(2)

        # add some variables to meet unit test coverate rate
        self.cls_token = nn.Parameter(torch.ones(1))
        self.mask_token = nn.Parameter(torch.ones(1))
        self.pos_embed = nn.Parameter(torch.ones(1))
        self.stem_norm = nn.Parameter(torch.ones(1))
        self.downsample_norm0 = nn.BatchNorm2d(2)
        self.downsample_norm1 = nn.BatchNorm2d(2)
        self.downsample_norm2 = nn.BatchNorm2d(2)
        self.lin = nn.Parameter(torch.ones(1))
        self.lin.requires_grad = False
        self.downsample_layers = nn.ModuleList()
        for _ in range(4):
            stage = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=True))
            self.downsample_layers.append(stage)


class ToyBEiT(nn.Module):

    def __init__(self):
        super().__init__()
        # add some variables to meet unit test coverate rate
        self.cls_token = nn.Parameter(torch.ones(1))
        self.patch_embed = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList()
        for _ in range(3):
            layer = nn.Conv2d(3, 3, 1)
            self.layers.append(layer)


class ToyMAE(nn.Module):

    def __init__(self):
        super().__init__()
        # add some variables to meet unit test coverate rate
        self.cls_token = nn.Parameter(torch.ones(1))
        self.patch_embed = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList()
        for _ in range(3):
            layer = nn.Conv2d(3, 3, 1)
            self.layers.append(layer)


class ToySegmentor(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.decode_head = nn.Conv2d(2, 2, kernel_size=1, groups=2)


class PseudoDataParallel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.module = model


class ToyViT(nn.Module):

    def __init__(self):
        super().__init__()


def check_optimizer_lr_wd(optimizer, gt_lr_wd):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd
    param_groups = optimizer.param_groups
    print(param_groups)
    assert len(param_groups) == len(gt_lr_wd)
    for i, param_dict in enumerate(param_groups):
        assert param_dict['weight_decay'] == gt_lr_wd[i]['weight_decay']
        assert param_dict['lr_scale'] == gt_lr_wd[i]['lr_scale']
        assert param_dict['lr_scale'] == param_dict['lr']


def test_learning_rate_decay_optimizer_constructor():

    # Test lr wd for ConvNeXT
    backbone = ToyConvNeXt()
    model = PseudoDataParallel(ToySegmentor(backbone))
    # stagewise decay
    stagewise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='stage_wise', num_layers=6)
    optimizer_cfg = dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05)
    optim_wrapper_cfg = dict(
        type='OptimWrapper',
        optimizer=optimizer_cfg,
        paramwise_cfg=stagewise_paramwise_cfg,
        constructor='LearningRateDecayOptimizerConstructor')
    optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
    check_optimizer_lr_wd(optim_wrapper.optimizer,
                          expected_stage_wise_lr_wd_convnext)
    # layerwise decay
    layerwise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='layer_wise', num_layers=6)
    optim_wrapper_cfg = dict(
        type='OptimWrapper',
        optimizer=optimizer_cfg,
        paramwise_cfg=layerwise_paramwise_cfg,
        constructor='LearningRateDecayOptimizerConstructor')
    optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
    check_optimizer_lr_wd(optim_wrapper.optimizer,
                          expected_layer_wise_lr_wd_convnext)

    # Test lr wd for BEiT
    backbone = ToyBEiT()
    model = PseudoDataParallel(ToySegmentor(backbone))

    layerwise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='layer_wise', num_layers=3)
    optim_wrapper_cfg = dict(
        type='OptimWrapper',
        optimizer=optimizer_cfg,
        paramwise_cfg=layerwise_paramwise_cfg,
        constructor='LearningRateDecayOptimizerConstructor')
    optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
    check_optimizer_lr_wd(optim_wrapper.optimizer,
                          expected_layer_wise_wd_lr_beit)

    # Test invalidation of lr wd for Vit
    backbone = ToyViT()
    model = PseudoDataParallel(ToySegmentor(backbone))
    with pytest.raises(NotImplementedError):
        optim_constructor = LearningRateDecayOptimizerConstructor(
            optim_wrapper_cfg, layerwise_paramwise_cfg)
        optim_constructor(model)
    with pytest.raises(NotImplementedError):
        optim_constructor = LearningRateDecayOptimizerConstructor(
            optim_wrapper_cfg, stagewise_paramwise_cfg)
        optim_constructor(model)

    # Test lr wd for MAE
    backbone = ToyMAE()
    model = PseudoDataParallel(ToySegmentor(backbone))

    layerwise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='layer_wise', num_layers=3)
    optim_wrapper_cfg = dict(
        type='OptimWrapper',
        optimizer=optimizer_cfg,
        paramwise_cfg=layerwise_paramwise_cfg,
        constructor='LearningRateDecayOptimizerConstructor')
    optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
    check_optimizer_lr_wd(optim_wrapper.optimizer,
                          expected_layer_wise_wd_lr_beit)


def test_beit_layer_decay_optimizer_constructor():

    # paramwise_cfg with BEiTExampleModel
    backbone = ToyBEiT()
    model = PseudoDataParallel(ToySegmentor(backbone))
    paramwise_cfg = dict(layer_decay_rate=2, num_layers=3)
    optim_wrapper_cfg = dict(
        type='OptimWrapper',
        constructor='LayerDecayOptimizerConstructor',
        paramwise_cfg=paramwise_cfg,
        optimizer=dict(
            type='AdamW', lr=1, betas=(0.9, 0.999), weight_decay=0.05))
    optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
    # optimizer = optim_wrapper_builder(model)
    check_optimizer_lr_wd(optim_wrapper.optimizer,
                          expected_layer_wise_wd_lr_beit)
