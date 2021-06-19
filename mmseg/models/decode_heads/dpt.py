import math

import torch
import torch.nn as nn
from mmcv.cnn import (Conv2d, ConvModule, build_activation_layer,
                      build_norm_layer)

from mmseg.ops import resize
from ..builder import HEADS
from ..utils import _make_readout_ops
from .decode_head import BaseDecodeHead


class ViTPostProcessBlock(nn.Module):

    def __init__(self,
                 in_channels=768,
                 out_channels=[96, 192, 384, 768],
                 img_size=[384, 384],
                 readout_type='ignore',
                 start_index=1,
                 scale_factors=[4, 2, 1, 0.5]):
        super(ViTPostProcessBlock, self).__init__()

        self.readout_ops = _make_readout_ops(in_channels, out_channels,
                                             readout_type, start_index)

        self.unflatten_size = torch.Size(
            [img_size[0] // 16, img_size[1] // 16])

    def forward(self, inputs):
        assert len(inputs) == len(self.readout_ops)

        return inputs


class ResidualConvUnit(nn.Module):

    def __init__(self, in_channels, act_cfg=None, norm_cfg=None):
        super(ResidualConvUnit, self).__init__()
        self.channels = in_channels

        self.activation = build_activation_layer(act_cfg)
        self.bn = False if norm_cfg is None else True
        self.bias = not self.bn

        self.conv1 = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            bias=self.bias)

        self.conv2 = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            bias=self.bias)

        if self.bn:
            _, self.bn1 = build_norm_layer(norm_cfg, self.channels)
            _, self.bn2 = build_norm_layer(norm_cfg, self.channels)

    def forward(self, inputs):
        x = self.activation(inputs)
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)

        x = self.activation(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)

        return x + inputs


class FeatureFusionBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 act_cfg=None,
                 norm_cfg=None,
                 deconv=False,
                 expand=False,
                 align_corners=True):
        super(FeatureFusionBlock, self).__init__()

        self.in_channels = in_channels
        self.expand = expand
        self.deconv = deconv
        self.align_corners = align_corners

        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2

        self.out_conv = Conv2d(
            self.in_channels, self.out_channels, kernel_size=1)

        self.res_conv_unit1 = ResidualConvUnit(self.in_channels, act_cfg,
                                               norm_cfg)
        self.res_conv_unit2 = ResidualConvUnit(self.in_channels, act_cfg,
                                               norm_cfg)

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            x = x + self.res_conv_unit1(inputs[1])
        x = self.res_conv_unit2(x)
        x = resize(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=self.align_corners)
        return self.out_conv(x)


@HEADS.register_module()
class DPTHead(BaseDecodeHead):
    """Vision Transformers for Dense Prediction.

    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.

    Args:
    """

    def __init__(self,
                 img_size=[384, 384],
                 out_channels=[96, 192, 384, 768],
                 readout_type='ignore',
                 patch_start_index=1,
                 post_process_kernel_size=[4, 2, 1, 3],
                 post_process_strides=[4, 2, 1, 2],
                 post_process_paddings=[0, 0, 0, 1],
                 expand_channels=False,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 **kwards):
        super(DPTHead, self).__init__(**kwards)

        self.in_channels = self.in_channels
        self.out_channels = out_channels
        self.expand_channels = expand_channels
        self.post_process_block = ViTPostProcessBlock(
            self.channels, out_channels, img_size, readout_type,
            patch_start_index, post_process_kernel_size, post_process_strides,
            post_process_paddings)

        out_channels = [
            channel * math.pow(2, idx) if expand_channels else channel
            for idx, channel in enumerate(self.out_channels)
        ]
        self.convs = []
        for idx, channel in enumerate(self.out_channels):
            self.convs.append(
                Conv2d(
                    channel, self.out_channels[idx], kernel_size=3, padding=1))

        self.refinenet0 = FeatureFusionBlock(self.channels, act_cfg, norm_cfg)
        self.refinenet1 = FeatureFusionBlock(self.channels, act_cfg, norm_cfg)
        self.refinenet2 = FeatureFusionBlock(self.channels, act_cfg, norm_cfg)
        self.refinenet3 = FeatureFusionBlock(self.channels, act_cfg, norm_cfg)

        self.conv = ConvModule(
            self.channels, self.channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.post_process_block(x)

        x = [self.convs[idx](feature) for idx, feature in enumerate(x)]

        path_3 = self.refinenet3(x[3])
        path_2 = self.refinenet2(path_3, x[2])
        path_1 = self.refinenet1(path_2, x[1])
        path_0 = self.refinenet0(path_1, x[0])

        x = self.conv(path_0)
        output = self.cls_seg(x)
        return output
