# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmseg.core.layer_decay_optimizer_constructor import \
    LayerDecayOptimizerConstructor

layer_wise_gt_lst = [{
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
}]


class BEiTExampleModel(nn.Module):

    def __init__(self, depth):
        super().__init__()
        self.backbone = nn.ModuleList()

        # add some variables to meet unit test coverate rate
        self.backbone.cls_token = nn.Parameter(torch.ones(1))
        self.backbone.patch_embed = nn.Parameter(torch.ones(1))
        self.backbone.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Conv2d(3, 3, 1)
            self.backbone.layers.append(layer)


def check_beit_adamw_optimizer(optimizer, gt_lst):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == 1
    assert optimizer.defaults['weight_decay'] == 0.05
    param_groups = optimizer.param_groups
    # 1 layer (cls_token and patch_embed) + 3 layers * 2 (w, b) = 7 layers
    assert len(param_groups) == 7
    for i, param_dict in enumerate(param_groups):
        assert param_dict['weight_decay'] == gt_lst[i]['weight_decay']
        assert param_dict['lr_scale'] == gt_lst[i]['lr_scale']
        assert param_dict['lr_scale'] == param_dict['lr']


def test_beit_layer_decay_optimizer_constructor():

    # paramwise_cfg with ConvNeXtExampleModel
    model = BEiTExampleModel(depth=3)
    optimizer_cfg = dict(
        type='AdamW', lr=1, betas=(0.9, 0.999), weight_decay=0.05)
    paramwise_cfg = dict(num_layers=3, layer_decay_rate=2)
    optim_constructor = LayerDecayOptimizerConstructor(optimizer_cfg,
                                                       paramwise_cfg)
    optimizer = optim_constructor(model)
    check_beit_adamw_optimizer(optimizer, layer_wise_gt_lst)
