# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import InvertedResidualV3, make_divisible


@BACKBONES.register_module()
class EfficientNet(BaseModule):
    """EfficientNet backbone.

    This backbone is the implementation of
    EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    https://arxiv.org/abs/1905.11946

    References:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py

    Args:
    strides (Sequence[int], optional): Strides of the first block of each
        layer.
    frozen_stages (int): Stages to be frozen (all param fixed).
        Default: -1, which means not freezing any parameters.
    conv_cfg (dict): Config dict for convolution layer.
        Default: None, which means using conv2d.
    norm_cfg (dict): Config dict for normalization layer.
        Default: dict(type='BN').
    act_cfg (dict): Config dict for activation layer.
        Default: dict(type='Swish').
    with_cp (bool): Use checkpoint or not. Using checkpoint will save some
        memory while slowing down the training speed. Default: False.
    pretrained (str, optional): model pretrained path. Default: None
    norm_eval (bool): Whether to set norm layers to eval mode, namely,
        freeze running stats (mean and var). Note: Effect on Batch Norm
        and its variants only. Default: False.
    model_name (str, optional): mode name to get the scale parameters.
        Default: efficientnet-b0
    init_cfg (dict or list[dict], optional): Initialization config dict.
        Default: None
    """
    model_settings = {
        # name:            width, depth, res
        'efficientnet-b0': (1.0, 1.0, 224),
        'efficientnet-b1': (1.0, 1.1, 240),
        'efficientnet-b2': (1.1, 1.2, 260),
        'efficientnet-b3': (1.2, 1.4, 300),
        'efficientnet-b4': (1.4, 1.8, 380),
        'efficientnet-b5': (1.6, 2.2, 456),
        'efficientnet-b6': (1.8, 2.6, 528),
        'efficientnet-b7': (2.0, 3.1, 600),
        'efficientnet-b8': (2.2, 3.6, 672),
        'efficientnet-l2': (4.3, 5.3, 800),
    }

    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks.
    arch_settings = [[1, 16, 2], [6, 24, 2], [6, 40, 2],
                     [6, 80, 3], [6, 112, 3], [6, 192, 4],
                     [6, 320, 2]]

    def __init__(self,
                 strides=(1, 2, 2, 2, 1, 2, 1),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Swish'),
                 with_cp=False,
                 pretrained=None,
                 norm_eval=False,
                 model_name='efficientnet-b0',
                 init_cfg=None):
        super(EfficientNet, self).__init__(init_cfg)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.strides = strides

        width_factor, depth_factor, res = self.model_settings[model_name]

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.in_channels = make_divisible(32 * width_factor, 8)

        self._conv_stem = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self._avg_pool = nn.AdaptiveAvgPool2d(1)

        self._layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = self.strides[i]
            out_channels = make_divisible(channel * width_factor, 8)
            num_blocks = int(num_blocks * depth_factor)
            inverted_res_layer = self._make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.in_channels = out_channels
            self._layers.append(layer_name)

        self.in_channels = make_divisible(1280 * width_factor, 8)

        self._conv_head = ConvModule(
            in_channels=out_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def _make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        """Stack InvertedResidual blocks of MobileNetV3 to build a layer for EfficientNet.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block.
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.
        """
        layers = []
        for i in range(num_blocks):
            mid_channels = self.in_channels * expand_ratio
            # se_cfg: channels = self.in_channels, se_layer: out_channels = channels // ratio; to get the original imp
            se_cfg = dict(
                channels=mid_channels,
                ratio=4,
                act_cfg=(dict(type='ReLU'),
                         dict(type='HSigmoid', bias=3.0, divisor=6.0)))
            layers.append(
                InvertedResidualV3(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    mid_channels=mid_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    se_cfg=se_cfg,
                    with_expand_conv=(self.in_channels != mid_channels),
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        for i in range(self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self._conv_stem(x)

        for i, layer_name in enumerate(self._layers):
            layer = getattr(self, layer_name)
            x = layer(x)

        x = self._conv_head(x)
        x = self._avg_pool(x)
        return x

    def train(self, mode=True):
        super(EfficientNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
