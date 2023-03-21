# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class ConvHead(BaseDecodeHead):
    def __init__(self,
                num_convs=2,
                input_channels=128,
                output_channels=128,
                **kwargs):
        super(ConvHead, self).__init__(**kwargs)

        self.num_convs = num_convs
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels
        for i in range(self.num_convs):
            self.conv_layers.append(nn.Conv2d(in_channels, self.output_channels, kernel_size=3, padding=1))
            in_channels = self.output_channels


    def forward(self, inputs):
        """Forward function."""
        x = inputs[self.in_index]

        for conv_layer in self.conv_layers:
            x = nn.functional.relu(conv_layer(x))

        output  = self.cls_seg(x)

        return output

