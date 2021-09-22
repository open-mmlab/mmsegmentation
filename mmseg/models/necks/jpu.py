# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from mmseg.ops import resize
from ..builder import NECKS


@NECKS.register_module()
class JPU(BaseModule):
    """FastFCN: Rethinking Dilated Convolution in the Backbone
    for Semantic Segmentation.

    This Joint Pyramid Upsampling (JPU) neck is the implementation of
    `FastFCN <https://arxiv.org/abs/1903.11816>`_.

    Args:
        in_channels (Tuple[int], optional): The number of input channels
            for each convolution operations before upsampling.
            Default: (256, 512, 1024, 2048).
        out_channels (int): The number of output channels. Default: 512.
        dilations (tuple[int]): Dilation rate of each layer.
        out_indices (Tuple[int] | int, optional): Output from which stages.
            Default: (0, 1, 2, 3).
        align_corners (bool, optional): The align_corners argument of
            resize operation. Default: False.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels=(256, 512, 1024, 2048),
                 out_channels=512,
                 dilations=(1, 2, 4, 8),
                 out_indices=(0, 1, 2, 3),
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(JPU, self).__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, tuple)
        assert len(in_channels) == 4, 'Length of input channels \
                                           must be 4!'

        assert len(dilations) == 4, 'Length of dilations \
                                           must be 4!'

        assert out_channels == in_channels[1], 'Output channels must \
                                           be the same with in_channels[1]!'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.out_indices = out_indices

        # Note: Names of operations below are referenced from original paper.
        for i in range(3):
            conv_name = f'conv{i+3}'
            conv_layer = nn.Sequential(
                ConvModule(
                    self.in_channels[i - 3],
                    self.out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.add_module(conv_name, conv_layer)
        for i in range(4):
            dilation_name = f'dilation{i+1}'
            dilation_layer = nn.Sequential(
                DepthwiseSeparableConvModule(
                    in_channels=3 * self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilations[i],
                    dilation=dilations[i],
                    dw_norm_cfg=norm_cfg,
                    dw_act_cfg=None,
                    pw_norm_cfg=norm_cfg,
                    pw_act_cfg=act_cfg))
            self.add_module(dilation_name, dilation_layer)

    def forward(self, inputs):
        """Forward function."""
        x_8 = inputs[1]
        x_16 = inputs[2]
        x_32 = inputs[3]
        feats = [self.conv5(x_32), self.conv4(x_16), self.conv3(x_8)]

        _, _, h, w = feats[-1].size()
        feats[-2] = resize(
            feats[-2],
            size=(h, w),
            mode='bilinear',
            align_corners=self.align_corners)
        feats[-3] = resize(
            feats[-3],
            size=(h, w),
            mode='bilinear',
            align_corners=self.align_corners)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([
            self.dilation1(feat),
            self.dilation2(feat),
            self.dilation3(feat),
            self.dilation4(feat)
        ],
                         dim=1)

        outs = [inputs[0], inputs[1], inputs[2], feat]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
