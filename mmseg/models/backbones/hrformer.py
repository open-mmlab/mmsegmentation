# Copyright (c) OpenMMLab. All rights reserved.

import math

import torch
import torch.nn as nn
# from timm.models.layers import to_2tuple, trunc_normal_
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, trunc_normal_init)
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import BaseModule
from torch.nn.functional import pad

from ..builder import BACKBONES
from .hrnet import Bottleneck, HRModule, HRNet


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        with_rpe (bool, optional): If True, use relative position bias.
            Default: True.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 with_rpe=True,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        self.with_rpe = with_rpe
        if self.with_rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                    num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            Wh, Ww = self.window_size
            rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
            rel_position_index = rel_index_coords + rel_index_coords.T
            rel_position_index = rel_position_index.flip(1)
            self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_init(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (B*num_windows, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.with_rpe:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1],
                    -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)
