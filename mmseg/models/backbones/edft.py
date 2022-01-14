import math
import warnings
from numpy import append

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import (Conv2d, Scale, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.runner import BaseModule, ModuleList, Sequential, _load_checkpoint

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbedOld as PatchEmbed
from ..utils import nchw_to_nlc, nlc_to_nchw, SelfAttentionBlock
from ..utils import CBAM, SELayer

from .mit import MixVisionTransformer
from .twins import SVT


class DepthFusionModule1(MultiheadAttention):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super().__init__(
            embed_dims * 2,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)
        self.embed_dims = embed_dims
        self.gamma = Scale(0)
        self.norm = build_norm_layer(norm_cfg, embed_dims * 2)[1]

    def forward(self, color, depth):
        h, w = color.size(2), color.size(3)
        qkv = torch.cat([color, depth], dim=1)
        qkv = nchw_to_nlc(qkv)
        out = self.attn(query=qkv, key=qkv, value=qkv, need_weights=False)[0]
        out = self.gamma(out) + qkv
        color = nlc_to_nchw(self.norm(out), (h, w))[:, :self.embed_dims]
        return color


class DepthFusionModule2(SelfAttentionBlock):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 norm_cfg=dict(type='BN'),
                 weight=1.0):
        super(DepthFusionModule2, self).__init__(
            key_in_channels=embed_dims,
            query_in_channels=embed_dims,
            channels=embed_dims,
            out_channels=embed_dims,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            key_query_norm=False,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=True,
            with_out=False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        self.gamma = Scale(0)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.spatial_gap = nn.Conv2d(embed_dims, 1, kernel_size=1, bias=True)
        self.weight = weight

    def forward(self, x, d):
        """Forward function."""
        d = self.spatial_gap(d)
        out = super(DepthFusionModule2, self).forward(x, x, d, self.weight)

        out = self.gamma(out) + x
        return self.norm(out)


class DepthFusionModule3(CBAM):

    def __init__(self, embed_dims):
        super(DepthFusionModule3, self).__init__(embed_dims * 2)
        self.embed_dims = embed_dims
        self.gamma = Scale(0)

    def forward(self, color, depth):
        x = torch.cat([color, depth], dim=1)
        out = super(DepthFusionModule3, self).forward(x)[:, :self.embed_dims]
        out = self.gamma(out) + color
        return color


class DepthFusionModule4(SELayer):

    def __init__(self, embed_dims):
        super(DepthFusionModule4, self).__init__(embed_dims * 2)
        self.embed_dims = embed_dims
        self.gamma = Scale(0)

    def forward(self, color, depth):
        x = torch.cat([color, depth], dim=1)
        out = super(DepthFusionModule4, self).forward(x)[:, :self.embed_dims]
        out = self.gamma(out) + color
        return color


class DepthDownsample(BaseModule):

    def __init__(self,
                 in_channels,
                 embed_dims=64,
                 num_stages=4,
                 num_heads=[1, 2, 4, 8],
                 strides=[4, 2, 2, 2],
                 overlap=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None):
        super(DepthDownsample, self).__init__()
        assert (in_channels == 1)

        patch_sizes = [7, 3, 3, 3] if overlap else [4, 2, 2, 2]
        self.pretrained = pretrained
        self.num_heads = num_heads
        self.layers = ModuleList()
        for i in range(num_stages):
            embed_dims_i = embed_dims * self.num_heads[i]
            self.layers.append(
                PatchEmbed(
                    in_channels=in_channels,
                    embed_dims=embed_dims_i,
                    kernel_size=patch_sizes[i],
                    stride=strides[i],
                    padding=patch_sizes[i] // 2 if overlap else 0,
                    pad_to_patch_size=False,
                    norm_cfg=norm_cfg,
                    # concat=True
                ))
            in_channels = embed_dims_i

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.load_state_dict(state_dict, False)

    def forward(self, x):
        outs = []
        for downsample in self.layers:
            x, hw_shape = downsample(x)
            x = nlc_to_nchw(x, hw_shape)
            outs.append(x)
        return outs


@BACKBONES.register_module()
class EDFT(BaseModule):

    def __init__(self, in_channels, **kwargs):
        super(EDFT, self).__init__()
        assert (in_channels == 4)

        self.num_heads = kwargs["num_heads"]
        self.num_stages = kwargs["num_stages"]

        self.weight = kwargs["weight"]
        self.overlap = kwargs["overlap"]
        self.attention_type = kwargs["attention_type"]
        self.same_branch = kwargs["same_branch"]
        self.backbone = kwargs["backbone"]
        kwargs.pop("weight")
        kwargs.pop("overlap")
        kwargs.pop("attention_type")
        kwargs.pop("same_branch")
        kwargs.pop("backbone")

        if (self.backbone == "Segformer"):
            self.color = MixVisionTransformer(3, **kwargs)
            self.embed_dims = kwargs["embed_dims"]
        elif (self.backbone == "Twins_svt"):
            self.color = SVT(3, **kwargs)
            self.embed_dims = kwargs["embed_dims"][0]
            self.num_heads = self.num_heads / 2
        else:
            raise NotImplementedError("{} backbone is not supported".format(
                self.backbone))

        if self.same_branch:
            if (self.backbone == "Segformer"):
                self.depth = MixVisionTransformer(1, **kwargs)
            elif (self.backbone == "Twins_svt"):
                self.depth = SVT(1, **kwargs)
            else:
                raise NotImplementedError(
                    "{} backbone is not supported".format(self.backbone))
        else:
            self.depth = DepthDownsample(
                1,
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                overlap=self.overlap,
                pretrained=kwargs["pretrained"]
                if "pretrained" in kwargs.keys() else None)

        self.dfms = ModuleList()
        for i in range(self.num_stages):
            embed_dims_i = self.embed_dims * self.num_heads[i]
            if self.attention_type == 'dsa-concat':
                self.dfms.append(
                    DepthFusionModule1(embed_dims_i, self.num_heads[i]))
            elif self.attention_type == 'dsa-add':
                self.dfms.append(
                    DepthFusionModule2(
                        embed_dims_i, self.num_heads[i], weight=self.weight))
            elif self.attention_type == 'ca':
                self.dfms.append(DepthFusionModule4(embed_dims_i))
            elif self.attention_type == 'cbam':
                self.dfms.append(DepthFusionModule3(embed_dims_i))
            else:
                pass  # self.attention_type == 'none'  just add

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=.02)
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    constant_init(m.bias, 0)
        # load pretrained model if exists
        self.color.init_weights()
        # self.depth.init_weights()

    def forward(self, x):
        c = x[:, :3]
        d = x[:, 3:]
        c_outs = self.color(c)
        d_outs = self.depth(d)

        outs = []
        for i in range(self.num_stages):
            c, d = c_outs[i], d_outs[i]
            if (len(self.dfms) != 0):
                out = self.dfms[i](c, d)
                outs.append(out)
            else:
                outs.append(c_outs[i] + d_outs[i])
        return outs