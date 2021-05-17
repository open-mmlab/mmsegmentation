import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MLAModule(nn.Module):

    def __init__(self,
                 align_corners,
                 mla_channels=256,
                 mlahead_channels=128,
                 norm_cfg=None):
        super(MLAModule, self).__init__()
        self.align_corners = align_corners
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
            self.head2(mla_p2), (4 * mla_p2.shape[-2], 4 * mla_p2.shape[-1]),
            mode='bilinear',
            align_corners=self.align_corners)
        head3 = F.interpolate(
            self.head3(mla_p3), (4 * mla_p3.shape[-2], 4 * mla_p3.shape[-1]),
            mode='bilinear',
            align_corners=self.align_corners)
        head4 = F.interpolate(
            self.head4(mla_p4), (4 * mla_p4.shape[-2], 4 * mla_p4.shape[-1]),
            mode='bilinear',
            align_corners=self.align_corners)
        head5 = F.interpolate(
            self.head5(mla_p5), (4 * mla_p5.shape[-2], 4 * mla_p5.shape[-1]),
            mode='bilinear',
            align_corners=self.align_corners)
        return torch.cat([head2, head3, head4, head5], dim=1)


@HEADS.register_module()
class SETRMLAHead(BaseDecodeHead):
    """Multi level feature aggretation head of SETR.

    MLA head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`.

    Args:
        mla_channels (int): Channels of reshape-conv of multi-level feature
            aggregation. Default: 256.
        mlahead_channels (int): Channels of conv-conv-4x of multi-level feature
            aggregation. Default: 128.
        mla_align_corners (bool): Whether to use align_corners in MLAModule.
            Default: True.
    """

    def __init__(self,
                 mla_align_corners=True,
                 mla_channels=256,
                 mlahead_channels=128,
                 **kwargs):
        super(SETRMLAHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.mla_channels = mla_channels
        self.mlahead_channels = mlahead_channels

        self.mlahead = MLAModule(
            align_corners=mla_align_corners,
            mla_channels=self.mla_channels,
            mlahead_channels=self.mlahead_channels,
            norm_cfg=self.norm_cfg)
        self.conv_seg = nn.Conv2d(
            4 * self.mlahead_channels, self.num_classes, 3, padding=1)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        out = self.mlahead(*inputs)
        out = self.conv_seg(out)
        return out
