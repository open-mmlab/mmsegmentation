# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead


class SelfAttentionBlock(_SelfAttentionBlock):
    """Self-Attention Module.

    Args:
        in_channels (int): Input channels of key/query feature.
        channels (int): Output channels of key/query transform.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict | None): Config of activation layers.
    """

    def __init__(self, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=True,
            with_out=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.output_project = self.build_project(
            in_channels,
            in_channels,
            num_convs=1,
            use_conv_module=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        context = super(SelfAttentionBlock, self).forward(x, x)
        return self.output_project(context)


@HEADS.register_module()
class ISAHead(BaseDecodeHead):
    """Interlaced Sparse Self-Attention for Semantic Segmentation.

    This head is the implementation of `ISA
    <https://arxiv.org/abs/1907.12273>`_.

    Args:
        isa_channels (int): The channels of ISA Module.
        down_factor (tuple[int]): The local group size of ISA.
    """

    def __init__(self, isa_channels, down_factor=(8, 8), **kwargs):
        super(ISAHead, self).__init__(**kwargs)
        self.down_factor = down_factor

        self.in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.global_relation = SelfAttentionBlock(
            self.channels,
            isa_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.local_relation = SelfAttentionBlock(
            self.channels,
            isa_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.out_conv = ConvModule(
            self.channels * 2,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x_ = self._transform_inputs(inputs)
        x = self.in_conv(x_)
        residual = x

        n, c, h, w = x.size()
        loc_h, loc_w = self.down_factor  # size of local group in H- and W-axes
        glb_h, glb_w = math.ceil(h / loc_h), math.ceil(w / loc_w)
        pad_h, pad_w = glb_h * loc_h - h, glb_w * loc_w - w
        if pad_h > 0 or pad_w > 0:  # pad if the size is not divisible
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                       pad_h - pad_h // 2)
            x = F.pad(x, padding)

        # global relation
        x = x.view(n, c, glb_h, loc_h, glb_w, loc_w)
        # do permutation to gather global group
        x = x.permute(0, 3, 5, 1, 2, 4)  # (n, loc_h, loc_w, c, glb_h, glb_w)
        x = x.reshape(-1, c, glb_h, glb_w)
        # apply attention within each global group
        x = self.global_relation(x)  # (n * loc_h * loc_w, c, glb_h, glb_w)

        # local relation
        x = x.view(n, loc_h, loc_w, c, glb_h, glb_w)
        # do permutation to gather local group
        x = x.permute(0, 4, 5, 3, 1, 2)  # (n, glb_h, glb_w, c, loc_h, loc_w)
        x = x.reshape(-1, c, loc_h, loc_w)
        # apply attention within each local group
        x = self.local_relation(x)  # (n * glb_h * glb_w, c, loc_h, loc_w)

        # permute each pixel back to its original position
        x = x.view(n, glb_h, glb_w, c, loc_h, loc_w)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (n, c, glb_h, loc_h, glb_w, loc_w)
        x = x.reshape(n, c, glb_h * loc_h, glb_w * loc_w)
        if pad_h > 0 or pad_w > 0:  # remove padding
            x = x[:, :, pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w // 2 + w]

        x = self.out_conv(torch.cat([x, residual], dim=1))
        out = self.cls_seg(x)

        return out
