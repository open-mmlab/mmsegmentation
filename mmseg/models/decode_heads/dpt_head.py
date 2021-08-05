import math
import torch
import torch.nn as nn

from mmcv.cnn import  ConvModule
from mmcv.runner.base_module import BaseModule, ModuleList
from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead

class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, norm_cfg, act_cfg):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.conv1 = ConvModule(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            order=("act", "conv", "norm")
        )
        self.conv2 =ConvModule(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            order=("act", "conv", "norm")
        )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, groups=1, expand=False, norm_cfg=None, act_cfg=dict(type='ReLU'), align_corners=True):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.expand = expand
        self.groups = groups

        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = ConvModule(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )#parameter bias and groups  should be adjusted

        self.resConfUnit1 = ResidualConvUnit(features, self.norm_cfg, self.act_cfg)
        self.resConfUnit2 = ResidualConvUnit(features, self.norm_cfg, self.act_cfg)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        
        output = resize(
                output,
                scale_factor=2,
                mode='bilinear',
                align_corners=self.align_corners)
        output = self.out_conv(output)

        return output

@HEADS.register_module()
class DPTHead(BaseDecodeHead):
    """ Dense Prediction Transformer Head 
    A PyTorch implement of : `Vision Transformers for Dense Prediction`
        https://arxiv.org/abs/2103.13413
        
    Inspiration from
        https://github.com/intel-isl/DPT

    Args:

    """
    def __init__(self, groups=1, expand=False, **kwargs):
        super(DPTHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.groups = groups
        self.expand = expand
        # Simulated FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for index, in_channel in enumerate(self.in_channels):
            l_conv = ConvModule(
                in_channel,
                self.channels*(1<<index if self.expand else 1),
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.groups,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = FeatureFusionBlock(
                self.channels,
                groups=self.groups,
                expand=self.expand,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                align_corners=self.align_corners)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        x = self.fpn_convs[-1](laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            x = self.fpn_convs[i](x,laterals[i-1])

        output = self.fpn_bottleneck(x)
        output = self.cls_seg(output)
    
        return output


