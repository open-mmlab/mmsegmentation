import math

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MLAConv(nn.Module):

    def __init__(self, in_channels=1024, mla_channels=256, norm_cfg=None):
        super(MLAConv, self).__init__()
        self.mla_p2_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 1, bias=False),
            build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p3_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 1, bias=False),
            build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p4_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 1, bias=False),
            build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p5_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 1, bias=False),
            build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p2 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p3 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p4 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p5 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())

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


@HEADS.register_module()
class SETRMLAAUXHead(BaseDecodeHead):
    """Vision Transformer with support for patch or hybrid CNN input stage.

    The extra head of MLA head of `SETR
    <https://arxiv.org/pdf/2012.15840.pdf>`

    Args:
        img_size (tuple): Input image size. Default: (224, 224).
        embed_dim (int): embedding dimension. Default: 1024.
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        mla_channels (int): Channels of reshape-conv of multi-level feature
            aggregation. Default: 256.
        mla_select_index (int): Selection index of mla output. Default: -1.
    """

    def __init__(self,
                 img_size=(224, 224),
                 embed_dim=1024,
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 mla_channels=256,
                 mla_select_index=-1,
                 **kwargs):
        super(SETRMLAAUXHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            raise TypeError('img_size must be type of int or tuple')

        self.embed_dim = embed_dim
        self.mla_channels = mla_channels
        self.mla_select_index = mla_select_index

        # In order to build general vision transformer backbone, we have to
        # move MLA to decode head.
        self.norm = nn.ModuleList([
            build_norm_layer(norm_layer, self.embed_dim)[1]
            for i in range(len(self.in_index))
        ])

        self.mla = MLAConv(
            in_channels=self.embed_dim,
            mla_channels=self.mla_channels,
            norm_cfg=self.norm_cfg)

        # In order to implement same constructure as the setr offcial repo,
        # we use mla select index to replace in_index of aux head of  offcial
        # repo.
        self.mla_in_channel = self.in_channels[self.mla_select_index]
        self.aux = nn.Conv2d(
            mla_channels, self.num_classes, kernel_size=1, bias=False)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        # Convert from nchw to nlc
        for i in range(len(inputs)):
            x = inputs[i]
            if x.dim() == 3:
                x = self.norm[i](x)
            elif x.dim() == 4:
                n, c, h, w = x.shape
                x = x.reshape(n, c, h * w).transpose(2, 1)
                x = self.norm[i](x)
            else:
                raise NotImplementedError
            inputs[i] = x

        inputs = self.mla(*inputs)
        x = inputs[self.mla_select_index]
        x = self.aux(x)
        x = F.interpolate(
            x,
            size=self.img_size,
            mode='bilinear',
            align_corners=self.align_corners)
        return x
