import math
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.runner import BaseModule, ModuleList, Sequential, load_checkpoint

from ...utils import get_root_logger
from ..builder import BACKBONES


def nlc_to_nchw(tensor, H, W):
    assert len(tensor.shape) == 3
    B, _, C = tensor.shape
    return tensor.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(tensor):
    assert len(tensor.shape) == 4
    return tensor.flatten(2).transpose(1, 2).contiguous()


class PEConv(BaseModule):
    """Mix-FFN use 3x3 depth-wise Conv to provide positional encode
    information.

    Args:
        embed_dims (int): The channels of token after embedding.
        kernel_size (int): The kernel size of convolution operation.
            Default: 3.
        stride (int): The kernel slide move distance of one step.
            Default: 1.
    """

    def __init__(self, embed_dims, kernel_size=3, stride=1):
        super().__init__()
        self.conv = ConvModule(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True,
            groups=embed_dims)

    def forward(self, x, H, W):

        x = self.conv(x)

        return x


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        conv1x1 = partial(
            ConvModule,
            kernel_size=1,
            stride=1,
            bias=True,
            norm_cfg=None,
            act_cfg=None)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    conv1x1(
                        in_channels=in_channels,
                        out_channels=feedforward_channels),
                    PEConv(feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
        layers.append(
            conv1x1(
                in_channels=in_channels, out_channels=feedforward_channels))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, H, W, identity=None):
        out = nlc_to_nchw(x, H, W)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    """An implementation of Efficient Multi-head Attention of Segformer.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = ConvModule(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            _, self.norm = build_norm_layer(norm_cfg, embed_dims)

    def forward(self, x, H, W, identity=None):
        B, _, C = x.shape

        x_q = x
        if self.sr_ratio > 1:
            x_kv = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_kv = self.sr(x_kv).reshape(B, C, -1).permute(0, 2, 1)
            x_kv = self.norm(x_kv)

        if identity is None:
            identity = x_q

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Defalut: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
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
                 batch_first=True,
                 sr_ratio=1):
        super(TransformerEncoderLayer, self).__init__()

        _, self.norm1 = build_norm_layer(norm_cfg, embed_dims)

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        _, self.norm2 = build_norm_layer(norm_cfg, embed_dims)

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    def forward(self, x, H, W):
        x = self.attn(self.norm1(x), H, W, identity=x)
        x = self.ffn(self.norm2(x), H, W, identity=x)
        return x


class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding."""

    def __init__(self,
                 patch_size,
                 in_channels,
                 embed_dims,
                 stride,
                 norm_cfg=None):
        super().__init__()

        self.proj = ConvModule(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            act_cfg=None,
            norm_cfg=None)
        self.norm = build_norm_layer(norm_cfg, embed_dims)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = nchw_to_nlc(x)
        x = self.norm(x)

        return x, H, W


@BACKBONES.register_module()
class MixVisionTransformer(BaseModule):
    """The backbone of Segformer.

    A PyTorch implement of : `SegFormer: Simple and Efficient Design for
    Semantic Segmentation with Transformers` -
        https://arxiv.org/pdf/2105.15203.pdf
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 sr_ratios=[8, 4, 2, 1],
                 pretrained=None,
                 init_cfg=None):
        super().__init__()

        self.out_indices = out_indices
        self.init_cfg = init_cfg
        self.pretrained = pretrained

        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic depth decay rule

        cur = 0
        self.layers = ModuleList()
        for stage_id in range(len(num_layers)):
            patch_embed = OverlapPatchEmbed(
                patch_size=patch_sizes[stage_id],
                in_channels=in_channels,
                embed_dims=embed_dims[0],
                stride=strides[stage_id],
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims[stage_id],
                    num_heads=num_heads[stage_id],
                    feedforward_channels=mlp_ratios[stage_id] * embed_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + i],
                    num_fcs=2,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[stage_id])
                for i in range(num_layers[stage_id])
            ])
            _, norm = build_norm_layer(norm_cfg, embed_dims[stage_id])
            self.layers.append(ModuleList(patch_embed, layer, norm))
            cur += num_layers[stage_id]

    def init_weights(self):
        if self.pretrained is None:
            for m in self.modules:
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
                        constant_init(m.bias)
        elif isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self,
                self.pretrained,
                map_location='cpu',
                strict=False,
                logger=logger)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i, layer in enumerate(self.layers):
            x, H, W = layer[0](x)
            x = layer[1](x, H, W)
            x = layer[2](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(x)

        return outs
