import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import ConvModule
from ..registry import HEADS
from .decode_head import DecodeHead


@HEADS.register_module
class ASPPHead(DecodeHead):

    def __init__(self, dilations=(1, 12, 24, 36), **kwargs):
        super(ASPPHead, self).__init__(**kwargs)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = nn.ModuleList()
        for dilation in dilations:
            self.aspp_modules.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        x = inputs[self.in_index]
        aspp_outs = [
            F.interpolate(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=True)
        ]
        for aspp_module in self.aspp_modules:
            aspp_out = aspp_module(x)
            aspp_outs.append(aspp_out)
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        output = self.cls_seg(output)
        return output
