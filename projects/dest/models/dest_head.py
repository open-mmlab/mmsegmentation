# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class DESTHead(BaseDecodeHead):

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        self.fuse_in_channels = self.in_channels.copy()
        for i in range(num_inputs - 1):
            self.fuse_in_channels[i] += self.fuse_in_channels[i + 1]
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=1,
                    stride=1,
                    act_cfg=self.act_cfg))

        self.fuse_convs = nn.ModuleList()
        for i in range(num_inputs):
            self.fuse_convs.append(
                ConvModule(
                    in_channels=self.fuse_in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    act_cfg=self.act_cfg))

        self.upsample = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode=interpolate_mode))
        ] * len(self.in_channels))

    def forward(self, inputs):
        feat = None
        for idx in reversed(range(len(inputs))):
            x = self.convs[idx](inputs[idx])
            if idx != len(inputs) - 1:
                x = torch.concat([feat, x], dim=1)
            x = self.upsample[idx](x)
            feat = self.fuse_convs[idx](x)
        return self.cls_seg(feat)
