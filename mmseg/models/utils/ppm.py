# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class DAPPM(BaseModule):

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__()

        self.num_scales = num_scales
        self.unsample_mode = upsample_mode
        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg

        self.scales = ModuleList()
        for i in range(num_scales):
            if i == 0:
                self.scales.append(
                    ConvModule(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        **conv_cfg))
            elif i == num_scales - 1:
                self.scales.append(
                    Sequential(*[
                        nn.AdaptiveAvgPool2d((1, 1)),
                        ConvModule(
                            in_channels,
                            branch_channels,
                            kernel_size=1,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            **conv_cfg)
                    ]))
            else:
                self.scales.append(
                    Sequential(*[
                        nn.AvgPool2d(
                            kernel_size=kernel_sizes[i - 1],
                            stride=strides[i - 1],
                            padding=paddings[i - 1]),
                        ConvModule(
                            in_channels,
                            branch_channels,
                            kernel_size=1,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            **conv_cfg)
                    ]))

        self.processes = ModuleList()
        for i in range(num_scales - 1):
            self.processes.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg))

        self.compression = ConvModule(
            branch_channels * num_scales,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

        self.shortcut = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

    def forward(self, inputs: Tensor):
        feats = []
        feats.append(self.scales[0](inputs))

        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))

        return self.compression(torch.cat(feats,
                                          dim=1)) + self.shortcut(inputs)


@MODELS.register_module()
class PAPPM(DAPPM):

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__(in_channels, branch_channels, out_channels,
                         num_scales, kernel_sizes, strides, paddings, norm_cfg,
                         act_cfg, conv_cfg, upsample_mode)

        self.processes = ConvModule(
            self.branch_channels * (self.num_scales - 1),
            self.branch_channels * (self.num_scales - 1),
            kernel_size=3,
            padding=1,
            groups=self.num_scales - 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            **self.conv_cfg)

    def forward(self, inputs: Tensor):
        x_ = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode,
                align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(torch.cat(feats, dim=1))
        return self.compression(torch.cat([x_, scale_out],
                                          dim=1)) + self.shortcut(inputs)
