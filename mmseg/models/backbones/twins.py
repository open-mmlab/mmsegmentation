# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.models.backbones.mit import EfficientMultiheadAttention
from mmseg.registry import MODELS
from ..utils.embed import PatchEmbed


class GlobalSubsampledAttention(EfficientMultiheadAttention):
    """Global Sub-sampled Attention (Spatial Reduction Attention)

    This module is modified from EfficientMultiheadAttention，
    which is a module from mmseg.models.backbones.mit.py.
    Specifically, there is no difference between
    `GlobalSubsampledAttention` and `EfficientMultiheadAttention`,
    `GlobalSubsampledAttention` is built as a brand new class
    because it is renamed as `Global sub-sampled attention (GSA)`
    in paper.


    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dims)
            or (n, batch, embed_dims). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of GSA of PCPVT.
            Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 batch_first=True,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 init_cfg=None):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            dropout_layer=dropout_layer,
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio,
            init_cfg=init_cfg)


class GSAEncoderLayer(BaseModule):
    """Implements one encoder layer with GSA.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (float): Kernel_size of conv in Attention modules. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)[1]
        self.attn = GlobalSubsampledAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape, identity=0.))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class LocallyGroupedSelfAttention(BaseModule):
    """Locally-grouped Self Attention (LSA) module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        window_size(int): Window size of LSA. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 window_size=1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert embed_dims % num_heads == 0, f'dim {embed_dims} should be ' \
                                            f'divided by num_heads ' \
                                            f'{num_heads}.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.window_size = window_size

    def forward(self, x, hw_shape):
        b, n, c = x.shape
        h, w = hw_shape
        x = x.view(b, h, w, c)

        # pad feature maps to multiples of Local-groups
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # calculate attention mask for LSA
        Hp, Wp = x.shape[1:-1]
        _h, _w = Hp // self.window_size, Wp // self.window_size
        mask = torch.zeros((1, Hp, Wp), device=x.device)
        mask[:, -pad_b:, :].fill_(1)
        mask[:, :, -pad_r:].fill_(1)

        # [B, _h, _w, window_size, window_size, C]
        x = x.reshape(b, _h, self.window_size, _w, self.window_size,
                      c).transpose(2, 3)
        mask = mask.reshape(1, _h, self.window_size, _w,
                            self.window_size).transpose(2, 3).reshape(
                                1, _h * _w,
                                self.window_size * self.window_size)
        # [1, _h*_w, window_size*window_size, window_size*window_size]
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-1000.0)).masked_fill(
                                              attn_mask == 0, float(0.0))

        # [3, B, _w*_h, nhead, window_size*window_size, dim]
        qkv = self.qkv(x).reshape(b, _h * _w,
                                  self.window_size * self.window_size, 3,
                                  self.num_heads, c // self.num_heads).permute(
                                      3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [B, _h*_w, n_head, window_size*window_size, window_size*window_size]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask.unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(b, _h, _w, self.window_size,
                                                  self.window_size, c)
        x = attn.transpose(2, 3).reshape(b, _h * self.window_size,
                                         _w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LSAEncoderLayer(BaseModule):
    """Implements one encoder layer in Twins-SVT.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
           Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True
        qk_scale (float | None, optional): Override default qk scale of
           head_dim ** -0.5 if set. Default: None.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        window_size (int): Window size of LSA. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 qk_scale=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 window_size=1,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)[1]
        self.attn = LocallyGroupedSelfAttention(embed_dims, num_heads,
                                                qkv_bias, qk_scale,
                                                attn_drop_rate, drop_rate,
                                                window_size)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class ConditionalPositionEncoding(BaseModule):
    """The Conditional Position Encoding (CPE) module.

    The CPE is the implementation of 'Conditional Positional Encodings
    for Vision Transformers <https://arxiv.org/abs/2102.10882>'_.

    Args:
       in_channels (int): Number of input channels.
       embed_dims (int): The feature dimension. Default: 768.
       stride (int): Stride of conv layer. Default: 1.
    """

    def __init__(self, in_channels, embed_dims=768, stride=1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
            groups=embed_dims)
        self.stride = stride

    def forward(self, x, hw_shape):
        b, n, c = x.shape
        h, w = hw_shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(b, c, h, w)
        if self.stride == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


@MODELS.register_module()
class PCPVT(BaseModule):
    """The backbone of Twins-PCPVT.

    This backbone is the implementation of `Twins: Revisiting the Design
    of Spatial Attention in Vision Transformers
    <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (list): Embedding dimension. Default: [64, 128, 256, 512].
        patch_sizes (list): The patch sizes. Default: [4, 2, 2, 2].
        strides (list): The strides. Default: [4, 2, 2, 2].
        num_heads (int): Number of attention heads. Default: [1, 2, 4, 8].
        mlp_ratios (int): Ratio of mlp hidden dim to embedding dim.
            Default: [4, 4, 4, 4].
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool): Enable bias for qkv if True. Default: False.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): Stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        depths (list): Depths of each stage. Default [3, 4, 6, 3]
        sr_ratios (list): Kernel_size of conv in each Attn module in
            Transformer encoder layer. Default: [8, 4, 2, 1].
        norm_after_stage（bool): Add extra norm. Default False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 patch_sizes=[4, 2, 2, 2],
                 strides=[4, 2, 2, 2],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 norm_after_stage=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths

        # patch_embed
        self.patch_embeds = ModuleList()
        self.position_encoding_drops = ModuleList()
        self.layers = ModuleList()

        for i in range(len(depths)):
            self.patch_embeds.append(
                PatchEmbed(
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dims=embed_dims[i],
                    conv_type='Conv2d',
                    kernel_size=patch_sizes[i],
                    stride=strides[i],
                    padding='corner',
                    norm_cfg=norm_cfg))

            self.position_encoding_drops.append(nn.Dropout(p=drop_rate))

        self.position_encodings = ModuleList([
            ConditionalPositionEncoding(embed_dim, embed_dim)
            for embed_dim in embed_dims
        ])

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for k in range(len(depths)):
            _block = ModuleList([
                GSAEncoderLayer(
                    embed_dims=embed_dims[k],
                    num_heads=num_heads[k],
                    feedforward_channels=mlp_ratios[k] * embed_dims[k],
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur + i],
                    num_fcs=2,
                    qkv_bias=qkv_bias,
                    act_cfg=dict(type='GELU'),
                    norm_cfg=dict(type='LN'),
                    sr_ratio=sr_ratios[k]) for i in range(depths[k])
            ])
            self.layers.append(_block)
            cur += depths[k]

        self.norm_name, norm = build_norm_layer(
            norm_cfg, embed_dims[-1], postfix=1)

        self.out_indices = out_indices
        self.norm_after_stage = norm_after_stage
        if self.norm_after_stage:
            self.norm_list = ModuleList()
            for dim in embed_dims:
                self.norm_list.append(build_norm_layer(norm_cfg, dim)[1])

    def init_weights(self):
        if self.init_cfg is not None:
            super().init_weights()
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x):
        outputs = list()

        b = x.shape[0]

        for i in range(len(self.depths)):
            x, hw_shape = self.patch_embeds[i](x)
            h, w = hw_shape
            x = self.position_encoding_drops[i](x)
            for j, blk in enumerate(self.layers[i]):
                x = blk(x, hw_shape)
                if j == 0:
                    x = self.position_encodings[i](x, hw_shape)
            if self.norm_after_stage:
                x = self.norm_list[i](x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

            if i in self.out_indices:
                outputs.append(x)

        return tuple(outputs)


@MODELS.register_module()
class SVT(PCPVT):
    """The backbone of Twins-SVT.

    This backbone is the implementation of `Twins: Revisiting the Design
    of Spatial Attention in Vision Transformers
    <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (list): Embedding dimension. Default: [64, 128, 256, 512].
        patch_sizes (list): The patch sizes. Default: [4, 2, 2, 2].
        strides (list): The strides. Default: [4, 2, 2, 2].
        num_heads (int): Number of attention heads. Default: [1, 2, 4].
        mlp_ratios (int): Ratio of mlp hidden dim to embedding dim.
            Default: [4, 4, 4].
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool): Enable bias for qkv if True. Default: False.
        drop_rate (float): Dropout rate. Default 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Default 0.0
        drop_path_rate (float): Stochastic depth rate. Default 0.2.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        depths (list): Depths of each stage. Default [4, 4, 4].
        sr_ratios (list): Kernel_size of conv in each Attn module in
            Transformer encoder layer. Default: [4, 2, 1].
        windiow_sizes (list): Window size of LSA. Default: [7, 7, 7],
        input_features_slice（bool): Input features need slice. Default: False.
        norm_after_stage（bool): Add extra norm. Default False.
        strides (list): Strides in patch-Embedding modules. Default: (2, 2, 2)
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256],
                 patch_sizes=[4, 2, 2, 2],
                 strides=[4, 2, 2, 2],
                 num_heads=[1, 2, 4],
                 mlp_ratios=[4, 4, 4],
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_cfg=dict(type='LN'),
                 depths=[4, 4, 4],
                 sr_ratios=[4, 2, 1],
                 windiow_sizes=[7, 7, 7],
                 norm_after_stage=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(in_channels, embed_dims, patch_sizes, strides,
                         num_heads, mlp_ratios, out_indices, qkv_bias,
                         drop_rate, attn_drop_rate, drop_path_rate, norm_cfg,
                         depths, sr_ratios, norm_after_stage, pretrained,
                         init_cfg)
        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        for k in range(len(depths)):
            for i in range(depths[k]):
                if i % 2 == 0:
                    self.layers[k][i] = \
                        LSAEncoderLayer(
                            embed_dims=embed_dims[k],
                            num_heads=num_heads[k],
                            feedforward_channels=mlp_ratios[k] * embed_dims[k],
                            drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=dpr[sum(depths[:k])+i],
                            qkv_bias=qkv_bias,
                            window_size=windiow_sizes[k])
