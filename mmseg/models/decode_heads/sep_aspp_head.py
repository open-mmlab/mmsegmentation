import torch
import torch.nn as nn

from mmseg.ops import ConvModule, SeparableConvModule, resize
from ..registry import HEADS
from .aspp_head import ASPPHead, ASPPModule


class SepASPPModule(ASPPModule):

    def __init__(self, **kwargs):
        super(SepASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = SeparableConvModule(
                    self.in_channels,
                    self.channels,
                    dilation=dilation,
                    norm_cfg=self.norm_cfg,
                    relu_first=False)


@HEADS.register_module
class SepASPPHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation

        This head is the implementation of Separable ASPP Head
        in (https://arxiv.org/abs/1802.02611)
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(SepASPPHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = SepASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            SeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                norm_cfg=self.norm_cfg,
                relu_first=False),
            SeparableConvModule(
                self.channels,
                self.channels,
                norm_cfg=self.norm_cfg,
                relu_first=False))

    def forward(self, inputs, prev_out=None):
        if prev_out is not None:
            assert prev_out.size(1) == self.prev_channels
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output
