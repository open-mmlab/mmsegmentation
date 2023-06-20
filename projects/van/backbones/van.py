# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmseg.models.backbones.mscan import (MSCAN, MSCABlock,
                                          MSCASpatialAttention,
                                          OverlapPatchEmbed)
from mmseg.registry import MODELS


class VANAttentionModule(BaseModule):

    def __init__(self, in_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.conv_spatial = nn.Conv2d(
            in_channels,
            in_channels,
            7,
            stride=1,
            padding=9,
            groups=in_channels,
            dilation=3)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class VANSpatialAttention(MSCASpatialAttention):

    def __init__(self, in_channels, act_cfg=dict(type='GELU')):
        super().__init__(in_channels, act_cfg=act_cfg)
        self.spatial_gating_unit = VANAttentionModule(in_channels)


class VANBlock(MSCABlock):

    def __init__(self,
                 channels,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__(
            channels,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)
        self.attn = VANSpatialAttention(channels)


@MODELS.register_module()
class VAN(MSCAN):

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super(MSCAN, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        # stochastic depth decay rule
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_channels=in_channels if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                norm_cfg=norm_cfg)

            block = nn.ModuleList([
                VANBlock(
                    channels=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def init_weights(self):
        return super().init_weights()
