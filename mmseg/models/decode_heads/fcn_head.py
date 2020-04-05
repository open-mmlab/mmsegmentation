import torch
import torch.nn as nn

from mmseg.ops import ConvModule
from ..registry import HEADS
from .decode_head import DecodeHead


@HEADS.register_module
class FCNHead(DecodeHead):

    def __init__(self, num_convs=2, concat_input=True, **kwargs):
        assert num_convs > 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        super(FCNHead, self).__init__(**kwargs)
        self.convs = nn.ModuleList()
        self.convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            self.convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        x = inputs[self.in_index]
        output = x
        for conv in self.convs:
            output = conv(output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
