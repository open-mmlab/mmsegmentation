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
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw, SelfAttentionBlock

from .mit import MixVisionTransformer


class DepthAlignModule(BaseModule):

    def __init__(self, features):
        super(DepthAlignModule, self).__init__()

        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False))

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False))

        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[h / s,
                                w / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, color, depth):
        h, w = color.size(2), color.size(3)

        concat = torch.cat((color, depth), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        color = self.bilinear_interpolate_torch_gridsample(
            color, (h, w), delta1)
        depth = self.bilinear_interpolate_torch_gridsample(
            depth, (h, w), delta2)

        return color, depth


class DepthFusionModule(MultiheadAttention):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN')):
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

    def forward(self, color, depth):
        h, w = color.size(2), color.size(3)
        qkv = torch.cat([color, depth], dim=1)
        qkv = nchw_to_nlc(qkv)
        out = self.attn(query=qkv, key=qkv, value=qkv, need_weights=False)[0]
        out = self.gamma(out) + qkv
        color = nlc_to_nchw(out, (h, w))[:, :self.embed_dims]
        return color


class DepthFusionModule2(SelfAttentionBlock):

    def __init__(self, embed_dims, num_heads):
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
            matmul_norm=False,
            with_out=False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        self.gamma = Scale(0)
        self.spatial_gap = nn.Conv2d(embed_dims, 1, kernel_size=1, bias=True)

    def forward(self, x, d):
        """Forward function."""
        d = self.spatial_gap(d)
        out = super(DepthFusionModule2, self).forward(x, x, d)

        out = self.gamma(out) + x
        return out


class DepthDownsample(BaseModule):

    def __init__(self,
                 in_channels,
                 embed_dims=64,
                 num_stages=4,
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None):
        super(DepthDownsample, self).__init__()
        assert (in_channels == 1)

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
                    padding=patch_sizes[i] // 2,
                    pad_to_patch_size=False,
                    norm_cfg=norm_cfg))
            in_channels = embed_dims_i

    def init_weights(self):
        if self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m.weight, 0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        constant_init(m.bias, 0)
        elif isinstance(self.pretrained, str):
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
            x, H, W = downsample(x), downsample.DH, downsample.DW
            x = nlc_to_nchw(x, (H, W))
            outs.append(x)
        return outs


@BACKBONES.register_module()
class MitFuse(BaseModule):

    def __init__(self, in_channels, **kwargs):
        super(MitFuse, self).__init__()
        assert (in_channels == 4)

        self.num_heads = kwargs["num_heads"]
        self.num_stages = kwargs["num_stages"]
        self.color = MixVisionTransformer(3, **kwargs)
        self.depth = MixVisionTransformer(1, **kwargs)
        # self.depth = DepthDownsample(
        #     1,
        #     embed_dims=kwargs["embed_dims"],
        #     num_heads=self.num_heads,
        #     pretrained=kwargs["pretrained"]
        #     if "pretrained" in kwargs.keys() else None)

        self.dams = ModuleList()
        self.dfms = ModuleList()
        for i in range(self.num_stages):
            embed_dims_i = kwargs["embed_dims"] * self.num_heads[i]
            # self.dams.append(DepthAlignModule(embed_dims_i))
            # self.dfms.append(
            #     DepthFusionModule2(embed_dims_i, self.num_heads[i]))

    def init_weights(self):
        self.color.init_weights()
        self.depth.init_weights()

    def forward(self, x):
        c = x[:, :3]
        d = x[:, 3:]
        c_outs = self.color(c)
        d_outs = self.depth(d)

        outs = []
        for i in range(self.num_stages):
            c, d = c_outs[i], d_outs[i]
            if (len(self.dams) != 0):
                c, d = self.dams[i](c, d)
            if (len(self.dfms) != 0):
                out = self.dfms[i](c, d)
                outs.append(out)
            else:
                outs.append(c_outs[i] + d_outs[i])
        return outs