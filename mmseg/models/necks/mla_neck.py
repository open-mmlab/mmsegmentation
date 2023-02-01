# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from mmseg.registry import MODELS


class MLAModule(nn.Module):

    def __init__(self,
                 in_channels=[1024, 1024, 1024, 1024],
                 out_channels=256,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        self.channel_proj = nn.ModuleList()
        for i in range(len(in_channels)):
            self.channel_proj.append(
                ConvModule(
                    in_channels=in_channels[i],
                    out_channels=out_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.feat_extract = nn.ModuleList()
        for i in range(len(in_channels)):
            self.feat_extract.append(
                ConvModule(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):

        # feat_list -> [p2, p3, p4, p5]
        feat_list = []
        for x, conv in zip(inputs, self.channel_proj):
            feat_list.append(conv(x))

        # feat_list -> [p5, p4, p3, p2]
        # mid_list -> [m5, m4, m3, m2]
        feat_list = feat_list[::-1]
        mid_list = []
        for feat in feat_list:
            if len(mid_list) == 0:
                mid_list.append(feat)
            else:
                mid_list.append(mid_list[-1] + feat)

        # mid_list -> [m5, m4, m3, m2]
        # out_list -> [o2, o3, o4, o5]
        out_list = []
        for mid, conv in zip(mid_list, self.feat_extract):
            out_list.append(conv(mid))

        return tuple(out_list)


@MODELS.register_module()
class MLANeck(nn.Module):
    """Multi-level Feature Aggregation.

    This neck is `The Multi-level Feature Aggregation construction of
    SETR <https://arxiv.org/abs/2012.15840>`_.


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
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # In order to build general vision transformer backbone, we have to
        # move MLA to neck.
        self.norm = nn.ModuleList([
            build_norm_layer(norm_layer, in_channels[i])[1]
            for i in range(len(in_channels))
        ])

        self.mla = MLAModule(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # Convert from nchw to nlc
        outs = []
        for i in range(len(inputs)):
            x = inputs[i]
            n, c, h, w = x.shape
            x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
            x = self.norm[i](x)
            x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
            outs.append(x)

        outs = self.mla(outs)
        return tuple(outs)
