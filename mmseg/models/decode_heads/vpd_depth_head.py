# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      build_upsample_layer)
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


class Decoder(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv,
                 num_filters,
                 deconv_kernels,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.deconv = num_deconv
        self.in_channels = in_channels

        self.deconv_layers = self._make_deconv_layer(
            num_deconv,
            num_filters,
            deconv_kernels,
        )

        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)

        # self.conv_layers = ConvModule(

        #     in_channels=num_filters[-1],
        #     out_channels=out_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     norm_cfg=dict(type='BN'),
        #     act_cfg=dict(type='ReLU')
        # )

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        out = self.deconv_layers(x)
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""

        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding


@MODELS.register_module()
class VPDDepthHead(BaseDecodeHead):
    num_classes = 1
    out_channels = 1

    def __init__(
        self,
        max_depth: float,
        in_channels: Sequence[int] = (320, 640, 1280, 1280),
        embed_dim: int = 192,
        feature_dim: int = 1024,
        num_deconv: int = 3,
        num_filters: Sequence[int] = (32, 32, 32),
        deconv_kernels: Sequence[int] = (2, 2, 2),
        fmap_border: Union[int, Sequence[int]] = 0,
        align_corners: bool = False,
        init_cfg=None,
    ):

        super(BaseDecodeHead, self).__init__(init_cfg=init_cfg)

        self.max_depth = max_depth
        self.align_corners = align_corners

        if isinstance(fmap_border, int):
            fmap_border = (fmap_border, fmap_border)
        self.fmap_border = fmap_border

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, in_channels[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels[0], in_channels[0], 3, stride=2, padding=1),
        )
        self.conv2 = nn.Conv2d(
            in_channels[1], in_channels[1], 3, stride=2, padding=1)

        self.conv_aggregation = nn.Sequential(
            nn.Conv2d(sum(in_channels), feature_dim, 1),
            nn.GroupNorm(16, feature_dim),
            nn.ReLU(),
        )

        self.decoder = Decoder(embed_dim * 8, embed_dim, num_deconv,
                               num_filters, deconv_kernels)

        self.depth_pred_layer = nn.Sequential(
            nn.Conv2d(
                embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(embed_dim, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = [
            x[0], x[1],
            torch.cat([x[2], F.interpolate(x[3], scale_factor=2)], dim=1)
        ]
        x = torch.cat([self.conv1(x[0]), self.conv2(x[1]), x[2]], dim=1)
        x = self.conv_aggregation(x)

        x = x[:, :, :x.size(2) - self.fmap_border[0], :x.size(3) -
              self.fmap_border[1]]
        x = self.decoder(x)
        out = self.depth_pred_layer(x)
        depth = torch.sigmoid(out) * self.max_depth

        return depth
