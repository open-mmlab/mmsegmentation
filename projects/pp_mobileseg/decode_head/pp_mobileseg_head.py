# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class PPMobileSegHead(nn.Module):
    """the segmentation head.

    Args:
        num_classes (int): the classes num.
        in_channels (int): the input channels.
        use_dw (bool): if to use deepwith convolution.
        dropout_ratio (float): Probability of an element to be zeroed.
            Default 0.0。
        align_corners (bool, optional): Geometrically, we consider the pixels
            of the input and output as squares rather than points.
        upsample (str): the upsample method.
        out_channels (int): the output channel.
        conv_cfg (dict): Config dict for convolution layer.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 use_dw=True,
                 dropout_ratio=0.1,
                 align_corners=False,
                 upsample='intepolate',
                 out_channels=None,
                 conv_cfg=dict(type='Conv'),
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN')):

        super().__init__()
        self.align_corners = align_corners
        self.last_channels = in_channels
        self.upsample = upsample
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.linear_fuse = ConvModule(
            in_channels=self.last_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            bias=False,
            groups=self.last_channels if use_dw else 1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = build_conv_layer(
            conv_cfg, self.last_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        x, x_hw = x[0], x[1]
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        if self.upsample == 'intepolate' or self.training or \
                self.num_classes < 30:
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)
        elif self.upsample == 'vim':
            labelset = torch.unique(torch.argmax(x, 1))
            x = torch.gather(x, 1, labelset)
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)

            pred = torch.argmax(x, 1)
            pred_retrieve = torch.zeros(pred.shape, dtype=torch.int32)
            for i, val in enumerate(labelset):
                pred_retrieve[pred == i] = labelset[i].cast('int32')

            x = pred_retrieve
        else:
            raise NotImplementedError(self.upsample, ' is not implemented')

        return [x]

    def predict(self, inputs, batch_img_metas: List[dict], test_cfg,
                **kwargs) -> List[Tensor]:
        """Forward function for testing, only ``pam_cam`` is used."""
        seg_logits = self.forward(inputs)[0]
        return seg_logits
