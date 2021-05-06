import math

import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from ..builder import NECKS


class MLAConv(nn.Module):

    def __init__(self,
                 in_channels=[1024, 1024, 1024, 1024],
                 mla_channels=256,
                 norm_cfg=None,
                 act_cfg=None):
        super(MLAConv, self).__init__()
        self.mla_p2_1x1 = ConvModule(
            in_channels[0],
            mla_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mla_p3_1x1 = ConvModule(
            in_channels[1],
            mla_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mla_p4_1x1 = ConvModule(
            in_channels[2],
            mla_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mla_p5_1x1 = ConvModule(
            in_channels[3],
            mla_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mla_p2 = ConvModule(
            mla_channels,
            mla_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mla_p3 = ConvModule(
            mla_channels,
            mla_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mla_p4 = ConvModule(
            mla_channels,
            mla_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mla_p5 = ConvModule(
            mla_channels,
            mla_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, res2, res3, res4, res5):

        res2 = self.to_2D(res2)
        res3 = self.to_2D(res3)
        res4 = self.to_2D(res4)
        res5 = self.to_2D(res5)

        mla_p5_1x1 = self.mla_p5_1x1(res5)
        mla_p4_1x1 = self.mla_p4_1x1(res4)
        mla_p3_1x1 = self.mla_p3_1x1(res3)
        mla_p2_1x1 = self.mla_p2_1x1(res2)

        mla_p4_plus = mla_p5_1x1 + mla_p4_1x1
        mla_p3_plus = mla_p4_plus + mla_p3_1x1
        mla_p2_plus = mla_p3_plus + mla_p2_1x1

        mla_p5 = self.mla_p5(mla_p5_1x1)
        mla_p4 = self.mla_p4(mla_p4_plus)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p2 = self.mla_p2(mla_p2_plus)

        return mla_p2, mla_p3, mla_p4, mla_p5


@NECKS.register_module()
class MLA(nn.Module):
    """Multi-level Feature Aggregation.

    The Multi-level Feature Aggregation construction of SETR:
    https://arxiv.org/pdf/2012.15840.pdf


    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 norm_cfg=None,
                 act_cfg=None):
        super(MLA, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # In order to build general vision transformer backbone, we have to
        # move MLA to neck.
        self.norm = nn.ModuleList([
            build_norm_layer(norm_layer, in_channels[i])[1]
            for i in range(len(in_channels))
        ])

        self.mla = MLAConv(
            in_channels=in_channels,
            mla_channels=out_channels,
            norm_cfg=norm_cfg)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        [print(x.shape) for x in inputs]

        # Convert from nchw to nlc
        outs = []
        for i in range(len(inputs)):
            x = inputs[i]
            if x.dim() == 3:
                x = self.norm[i](x)
            elif x.dim() == 4:
                n, c, h, w = x.shape
                x = x.reshape(n, c, h * w).transpose(2, 1)
                x = self.norm[i](x)
            outs.append(x)

        outs = self.mla(*outs)
        return tuple(outs)
