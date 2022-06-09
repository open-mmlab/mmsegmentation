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
            Default: (512, 1024, 2048).
        mid_channels (int): The number of output channels of JPU.
            Default: 512.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        dilations (tuple[int]): Dilation rate of each Depthwise
            Separable ConvModule. Default: (1, 2, 4, 8).
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
                 in_channels=(512, 1024, 2048),
                 mid_channels=512,
                 start_level=0,
                 end_level=-1,
                 dilations=(1, 2, 4, 8),
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(JPU, self).__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, tuple)
        assert isinstance(dilations, tuple)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.start_level = start_level
        self.num_ins = len(in_channels)
        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)

        self.dilations = dilations
        self.align_corners = align_corners

        self.conv_layers = nn.ModuleList()
        self.dilation_layers = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            conv_layer = nn.Sequential(
                ConvModule(
                    self.in_channels[i],
                    self.mid_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.conv_layers.append(conv_layer)
        for i in range(len(dilations)):
            dilation_layer = nn.Sequential(
                DepthwiseSeparableConvModule(
                    in_channels=(self.backbone_end_level - self.start_level) *
                    self.mid_channels,
                    out_channels=self.mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilations[i],
                    dilation=dilations[i],
                    dw_norm_cfg=norm_cfg,
                    dw_act_cfg=None,
                    pw_norm_cfg=norm_cfg,
                    pw_act_cfg=act_cfg))
            self.dilation_layers.append(dilation_layer)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels), 'Length of inputs must \
                                           be the same with self.in_channels!'

        feats = [
            self.conv_layers[i - self.start_level](inputs[i])
            for i in range(self.start_level, self.backbone_end_level)
        ]

        h, w = feats[0].shape[2:]
        for i in range(1, len(feats)):
            feats[i] = resize(
                feats[i],
                size=(h, w),
                mode='bilinear',
                align_corners=self.align_corners)

        feat = torch.cat(feats, dim=1)
        concat_feat = torch.cat([
            self.dilation_layers[i](feat) for i in range(len(self.dilations))
        ],
                                dim=1)

        outs = []

        # Default: outs[2] is the output of JPU for decoder head, outs[1] is
        # the feature map from backbone for auxiliary head. Additionally,
        # outs[0] can also be used for auxiliary head.
        for i in range(self.start_level, self.backbone_end_level - 1):
            outs.append(inputs[i])
        outs.append(concat_feat)
        return tuple(outs)
