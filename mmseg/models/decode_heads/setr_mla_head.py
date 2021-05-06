import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MLAHead(nn.Module):

    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(
            nn.Conv2d(
                mla_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(),
            nn.Conv2d(
                mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head3 = nn.Sequential(
            nn.Conv2d(
                mla_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(),
            nn.Conv2d(
                mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head4 = nn.Sequential(
            nn.Conv2d(
                mla_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(),
            nn.Conv2d(
                mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head5 = nn.Sequential(
            nn.Conv2d(
                mla_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(),
            nn.Conv2d(
                mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = F.interpolate(
            self.head2(mla_p2),
            4 * mla_p2.shape[-1],
            mode='bilinear',
            align_corners=True)
        head3 = F.interpolate(
            self.head3(mla_p3),
            4 * mla_p3.shape[-1],
            mode='bilinear',
            align_corners=True)
        head4 = F.interpolate(
            self.head4(mla_p4),
            4 * mla_p4.shape[-1],
            mode='bilinear',
            align_corners=True)
        head5 = F.interpolate(
            self.head5(mla_p5),
            4 * mla_p5.shape[-1],
            mode='bilinear',
            align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)


@HEADS.register_module()
class SETRMLAHead(BaseDecodeHead):
    """Vision Transformer with support for patch or hybrid CNN input stage.

    MLA head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`.

    Args:
        img_size (tuple): Input image size. Default: (384, 384).
        embed_dim (int): embedding dimension. Default: 1024.
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        mla_channels (int): Channels of reshape-conv of multi-level feature
            aggregation. Default: 256.
        mlahead_channels (int): Channels of conv-conv-4x of multi-level feature
            aggregation. Default: 128.
    """

    def __init__(self,
                 img_size=(384, 384),
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 mla_channels=256,
                 mlahead_channels=128,
                 **kwargs):
        super(SETRMLAHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.img_size = img_size
        self.mla_channels = mla_channels
        self.mlahead_channels = mlahead_channels

        self.mlahead = MLAHead(
            mla_channels=self.mla_channels,
            mlahead_channels=self.mlahead_channels,
            norm_cfg=self.norm_cfg)
        self.conv_seg = nn.Conv2d(
            4 * self.mlahead_channels, self.num_classes, 3, padding=1)

    def to_1D(self, x):

        def convertion(x):
            n, c, h, w = x.shape
            x = x.reshape(n, c, h * w).transpose(2, 1)
            return x

        if isinstance(x, list) or isinstance(x, tuple):
            outs = []
            for item in x:
                outs.append(convertion(item))
            x = outs
        else:
            x = convertion(x)
        return x

    def to_2D(self, x):

        def convertion(x):
            n, hw, c = x.shape
            h = w = int(math.sqrt(hw))
            x = x.transpose(1, 2).reshape(n, c, h, w)
            return x

        if isinstance(x, list) or isinstance(x, tuple):
            outs = []
            for item in x:
                outs.append(convertion(item))
            x = outs
        else:
            x = convertion(x)
        return x

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        if inputs[0].dim() == 3:
            inputs = self.to_2D(inputs)

        out = self.mlahead(*inputs)
        out = self.conv_seg(out)

        out = F.interpolate(
            out,
            size=self.img_size,
            mode='bilinear',
            align_corners=self.align_corners)
        return out
