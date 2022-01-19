# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule

from mmseg.models.builder import BACKBONES


class Block(BaseModule):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) ->
    1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C);
    LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float): Init value for Layer Scale.
            Default: 1e-6.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
    """

    def __init__(self, dim, layer_scale_init_value=1e-6, dropout_layer=None):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim,
            4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = build_dropout(dropout_layer)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


@BACKBONES.register_module()
class ConvNeXt(BaseModule):
    """The backbone of ConvNext.

    This backbone is the implementation of `A ConvNet for the 2020s
    <https://arxiv.org/abs/2201.03545>`_.

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        depths (tuple(int)): Number of blocks at each stage.
            Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage.
            Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate.
            Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Default: 1e-6.
        out_indices (List[int] | int, optional): Output from which stages.
            Default: [0, 1, 2, 3].
        pretrained (str, optional): Model pretrained path.
            Default: None
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_chans=3,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 pretrained=None,
                 init_cfg=None):
        super(ConvNeXt, self).__init__(init_cfg=init_cfg)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.downsample_layers = nn.ModuleList(
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4))
        self.stem_norm = nn.LayerNorm(dims[0], eps=1e-6)
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Conv2d(
                dims[i], dims[i + 1], kernel_size=2, stride=2)
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages,
        # each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                Block(
                    dim=dims[i],
                    layer_scale_init_value=layer_scale_init_value,
                    dropout_layer=dict(
                        type='DropPath', drop_prob=dp_rates[cur + j]))
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        for i_downsample_layer in range(3):
            layer = norm_layer(dims[i_downsample_layer])
            layer_name = f'downsample_norm{i_downsample_layer}'
            self.add_module(layer_name, layer)

    def init_weights(self, pretrained=None):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            super(ConvNeXt, self).init_weights()
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        for i in range(4):
            if i != 0:
                downsample_norm = getattr(self, f'downsample_norm{i-1}')
                x = downsample_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = self.downsample_layers[i](x)
            if i == 0:
                x = self.stem_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                outs.append(x_out)
        return tuple(outs)
