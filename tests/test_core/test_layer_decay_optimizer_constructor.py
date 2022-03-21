# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmseg.core.layer_decay_optimizer_constructor import \
    LayerDecayOptimizerConstructor


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


def test_beit_layer_decay_optimizer_constructor():

    # paramwise_cfg with ConvNeXtExampleModel
    model = BEiTExampleModel(depth=12)
    optimizer_cfg = dict(
        type='AdamW', lr=1, betas=(0.9, 0.999), weight_decay=0.05)
    paramwise_cfg = dict(num_layers=12, layer_decay_rate=0.9)
    optim_constructor = LayerDecayOptimizerConstructor(optimizer_cfg,
                                                       paramwise_cfg)
    optim_constructor(model)
