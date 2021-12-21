# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner import BaseModule, _load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from torch.nn import functional as F
from torch.nn.functional import dropout, linear, pad, softmax

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils import nchw_to_nlc, nlc_to_nchw
from .resnet import Bottleneck


def build_drop_path(drop_path_rate):
    return build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))


class MultiheadAttention(BaseModule):
    """MultiheadSelfAttention module with relative position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        dropout (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        kdim (int, optional): The number of channels of q/k.
            Default: None
        kdim (int, optional): The number of channels of v.
            Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 window_size=(7, 7),
                 bias=True,
                 dropout=0.0,
                 kdim=None,
                 vdim=None,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim
                ), 'embed_dim must be divisible by num_heads'

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.window_size = [window_size] * 2
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # pairwise relative position for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :,
                        0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index',
                             relative_position_index)

    def init_weights(self):
        trunc_normal_init(self.relative_position_bias_table, std=0.02)

    def forward(self, query, key, value):
        return self.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            dropout_p=self.dropout,
            training=self.training,
            out_dim=self.vdim)

    def multi_head_attention_forward(self,
                                     query,
                                     key,
                                     value,
                                     embed_dim_to_check,
                                     num_heads,
                                     dropout_p=0.0,
                                     out_dim=None,
                                     training=True):
        tgt_len, bsz, embed_dim = query.size()
        key = query if key is None else key
        value = query if value is None else value

        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // self.num_heads
        out_dim = self.vdim
        num_heads = self.num_heads
        v_head_dim = self.vdim // num_heads
        assert (head_dim * self.num_heads == embed_dim
                ), 'embed_dim must be divisible by num_heads'
        scaling = float(head_dim)**-0.5

        # whether or not use the original query/key/value
        q = self.q_proj(query) * scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        # divide heads
        q = q.contiguous().view(tgt_len, bsz * num_heads,
                                head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads,
                                    head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads,
                                    v_head_dim).transpose(0, 1)
        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(
            attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
        """
        Add relative position embedding
        """
        # NOTE: we assume that the src_len == tgt_len == window_size**2
        assert (
            src_len == self.window_size[0] * self.window_size[1]
            and tgt_len == self.window_size[0] * self.window_size[1]
        ), f'src_len={src_len}, tgt_len={tgt_len},' \
            f' window_size={self.window_size}'
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len,
            src_len) + relative_position_bias.unsqueeze(0)
        attn_output_weights = attn_output_weights.view(bsz * num_heads,
                                                       tgt_len, src_len)
        """
        Reweight the attention map before softmax().
        attn_output_weights: (b*n_head, n, hw)
        """
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(
            attn_output_weights, p=dropout_p, training=training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(
            attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
        attn_output = (
            attn_output.transpose(0,
                                  1).contiguous().view(tgt_len, bsz, out_dim))
        attn_output = linear(attn_output, self.out_proj.weight,
                             self.out_proj.bias)
        return attn_output


class LocalWindowSelfAttention(BaseModule):
    r""" Local-window Self Attention (LSA) module with relative position bias.

    This module is the short-range self-attention module in the
    Interlaced Sparse Self-Attention <https://arxiv.org/abs/1907.12273>`_.

    Args:
        window_size (tuple[int]): Window size.
    """

    def __init__(self, *args, window_size=7, init_cfg=None, **kwargs):
        super(LocalWindowSelfAttention, self).__init__(init_cfg=init_cfg)

        self.window_size = window_size
        self.attn = MultiheadAttention(
            *args, window_size=window_size, init_cfg=None, **kwargs)

    def forward(self, x, H, W, **kwargs):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        Wh = Ww = self.window_size

        # center-pad the feature on H and W axes
        pad_h = math.ceil(H / Wh) * Wh - H
        pad_w = math.ceil(W / Ww) * Ww - W
        x = pad(x, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2))

        # permute
        x = x.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C)
        x = x.permute(2, 4, 0, 1, 3, 5)
        x = x.reshape(Wh * Ww, -1, C)

        # attention
        out = self.attn(x, x, x, **kwargs)

        # reverse permutation
        out = out.reshape(Wh, Ww, B, math.ceil(H / Wh), math.ceil(W / Ww), C)
        out = out.permute(2, 0, 3, 1, 4, 5)
        out = out.reshape(B, H + pad_h, W + pad_w, C)

        # de-pad
        out = out[:, pad_h // 2:H + pad_h // 2, pad_w // 2:W + pad_w // 2]
        return out.reshape(B, N, C)


class CrossFFN(BaseModule):
    """FFN with Depthwise Conv of HRFormer.

    Args:
        in_features (int): The feature dimension.
        hidden_features (int, optional): The hidden dimension of FFNs.
            Defaults: The same as in_features.
        act_layer (nn.Module, optional): The activation for 1x1 convs.
            Default: nn.GELU
        dw_act_layer (nn.Module, optional): The activation for 3x3 dw convs.
            Default: nn.GELU
        norm_layer (nn.Module, optional): The normalization layer of FFNs.
            Default: nn.SyncBatchNorm
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=nn.GELU,
                 dw_act_cfg=nn.GELU,
                 norm_cfg=nn.SyncBatchNorm,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = build_activation_layer(act_cfg)
        self.norm1 = build_norm_layer(norm_cfg, hidden_features)[1]
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1)
        self.act2 = build_activation_layer(dw_act_cfg)
        self.norm2 = build_norm_layer(norm_cfg, hidden_features)[1]
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = build_activation_layer(act_cfg)
        self.norm3 = build_norm_layer(norm_cfg, out_features)[1]

        # put the modules togather
        self.layers = [
            self.fc1, self.norm1, self.act1, self.dw3x3, self.norm2, self.act2,
            self.fc2, self.norm3, self.act3
        ]

    def forward(self, x, H, W):
        x = nlc_to_nchw(x, (H, W))
        for layer in self.layers:
            x = layer(x)
        x = nchw_to_nlc(x)
        return x


class HRFormerBlock(BaseModule):
    """High-Resolution Block for HRFormer.

    Args:
        in_features (int): The input dimension.
        out_features (int): The output dimension.
        num_heads (int): The number of head within each LSA.
        window_size (int, optional): The window size for the LSA.
            Default: 7
        mlp_ratio (int, optional): The expansion ration of FFN.
            Default: 4
        act_layer (nn.Module, optional): The activation layer.
            Default: nn.GELU
        norm_layer (nn.Module, optional): The normalization layer.
            Default: nn.LayerNorm
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    expansion = 1

    def __init__(self,
                 in_features,
                 out_features,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.0,
                 drop_path=0.0,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN'),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None,
                 **kwargs):
        super(HRFormerBlock, self).__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = build_norm_layer(transformer_norm_cfg, in_features)[1]
        self.attn = LocalWindowSelfAttention(
            in_features,
            num_heads=num_heads,
            window_size=window_size,
            init_cfg=None,
            **kwargs)

        self.norm2 = build_norm_layer(transformer_norm_cfg, out_features)[1]
        self.ffn = CrossFFN(
            in_features=in_features,
            hidden_features=int(in_features * mlp_ratio),
            out_features=out_features,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dw_act_cfg=act_cfg,
            init_cfg=None)

        self.drop_path = build_drop_path(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.size()
        # Attention
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # FFN
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

    def extra_repr(self):
        # (Optional) Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'num_heads={}, window_size={}, mlp_ratio={}'.format(
            self.num_heads, self.window_size, self.mlp_ratio)


class HRFomerModule(BaseModule):
    """High-Resolution Module for HRFormer.

    Args:
        num_branches (int): The number of branches in the HRFormerModule.
        block (nn.Module): The building block of HRFormer.
            The block should be the HRFormerBlock.
        num_blocks (tuple): The number of blocks in each branch.
            The length must be equal to num_branches.
        num_inchannels (tuple): The number of input channels in each branch.
            The length must be equal to num_branches.
        num_channels (tuple): The number of channels in each branch.
            The length must be equal to num_branches.
        num_heads (tuple): The number of heads within the LSAs.
        num_window_sizes (tuple): The window size for the LSAs.
        num_mlp_ratios (tuple): The expansion ratio for the FFNs.
        drop_path (int, optional): The drop path rate of HRFomer.
            Default: 0.0
        multiscale_output (bool, optional): Whether to output multi-level
            features produced by multiple branches. If False, only the first
            level feature will be output. Default: True.
        conv_cfg (dict, optional): Config of the conv layers.
            Default: None.
        norm_cfg (dict, optional): Config of the norm layers appended
            right after conv. Default: None.
        transformer_norm_cfg (dict, optional): Config of the norm layers.
            Default: None.
    """

    def __init__(self,
                 num_branches,
                 block,
                 num_blocks,
                 num_inchannels,
                 num_channels,
                 num_heads,
                 num_window_sizes,
                 num_mlp_ratios,
                 multiscale_output=True,
                 drop_path=0.0,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):

        super(HRFomerModule, self).__init__(init_cfg=init_cfg)
        self._check_branches(num_branches, num_blocks, num_inchannels,
                             num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.transformer_norm_cfg = transformer_norm_cfg

        self.multiscale_output = multiscale_output
        self.branches = self._make_branches(num_branches, block, num_blocks,
                                            num_channels, num_heads,
                                            num_window_sizes, num_mlp_ratios,
                                            drop_path)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios

    def _check_branches(self, num_branches, num_blocks, num_inchannels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'num_branches({}) <> num_blocks({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'num_branches({}) <> num_channels({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'num_branches({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         num_heads, num_window_sizes, num_mlp_ratios,
                         drop_paths):
        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                num_heads=num_heads[branch_index],
                window_size=num_window_sizes[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                drop_path=drop_paths[0],
                norm_cfg=self.norm_cfg,
                transformer_norm_cfg=self.transformer_norm_cfg,
                init_cfg=None))

        self.num_inchannels[
            branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    drop_path=drop_paths[i],
                    norm_cfg=self.norm_cfg,
                    transformer_norm_cfg=self.transformer_norm_cfg,
                    init_cfg=None))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels,
                       num_heads, num_window_sizes, num_mlp_ratios,
                       drop_paths):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    drop_paths=drop_paths))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multiscale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_inchannels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            with_out_act = False
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            with_out_act = True
                        sub_modules = [
                            build_conv_layer(
                                self.conv_cfg,
                                num_inchannels[j],
                                num_inchannels[j],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=num_inchannels[j],
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg,
                                             num_inchannels[j])[1],
                            build_conv_layer(
                                self.conv_cfg,
                                num_inchannels[j],
                                num_outchannels_conv3x3,
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg,
                                             num_outchannels_conv3x3)[1]
                        ]
                        if with_out_act:
                            sub_modules.append(nn.ReLU(False))
                        conv3x3s.append(nn.Sequential(*sub_modules))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


@BACKBONES.register_module()
class HRFormer(BaseModule):
    """HRFormer backbone.

    This backbone is the implementation of `HRFormer: High-Resolution
    Transformer for Dense Prediction <https://arxiv.org/abs/2110.09408>`_.

    Args:
        extra (dict): Detailed configuration for each stage of HRNet.
            There must be 4 stages, the configuration for each stage must have
            5 keys:

                - num_modules (int): The number of HRModule in this stage.
                - num_branches (int): The number of branches in the HRModule.
                - block (str): The type of block.
                - num_blocks (tuple): The number of blocks in each branch.
                    The length must be equal to num_branches.
                - num_channels (tuple): The number of channels in each branch.
                    The length must be equal to num_branches.
        in_channels (int): Number of input image channels. Normally 3.
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): Config of norm layer.
            Use `SyncBN` by default.
        transformer_norm_cfg (dict): Config of transformer norm layer.
            Use `LN` by default.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: False.
        multiscale_output (bool): Whether to output multi-level features
            produced by multiple branches. If False, only the first level
            feature will be output. Default: True.
        pretrained (str, optional): Model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> from mmseg.models import HRFormer
        >>> import torch
        >>> extra = dict(
        >>>     stage1=dict(
        >>>         num_modules=1,
        >>>         num_branches=1,
        >>>         block='BOTTLENECK',
        >>>         num_blocks=(2, ),
        >>>         num_channels=(64, )),
        >>>     stage2=dict(
        >>>         num_modules=1,
        >>>         num_branches=2,
        >>>         block='HRFORMER',
        >>>         window_sizes=(7, 7),
        >>>         num_heads=(1, 2),
        >>>         mlp_ratios=(4, 4),
        >>>         num_blocks=(2, 2),
        >>>         num_channels=(32, 64)),
        >>>     stage3=dict(
        >>>         num_modules=4,
        >>>         num_branches=3,
        >>>         block='HRFORMER',
        >>>         window_sizes=(7, 7, 7),
        >>>         num_heads=(1, 2, 4),
        >>>         mlp_ratios=(4, 4, 4),
        >>>         num_blocks=(2, 2, 2),
        >>>         num_channels=(32, 64, 128)),
        >>>     stage4=dict(
        >>>         num_modules=2,
        >>>         num_branches=4,
        >>>         block='HRFORMER',
        >>>         window_sizes=(7, 7, 7, 7),
        >>>         num_heads=(1, 2, 4, 8),
        >>>         mlp_ratios=(4, 4, 4, 4),
        >>>         num_blocks=(2, 2, 2, 2),
        >>>         num_channels=(32, 64, 128, 256)))
        >>> self = HRFormer(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
        (1, 64, 4, 4)
        (1, 128, 2, 2)
        (1, 256, 1, 1)
    """

    blocks_dict = {'BOTTLENECK': Bottleneck, 'HRFORMER': HRFormerBlock}

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 norm_eval=False,
                 drop_path_rate=0.,
                 frozen_stages=-1,
                 multiscale_output=True,
                 pretrained=None,
                 init_cfg=None):
        super(HRFormer, self).__init__(init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        # Assert configurations of 4 stages are in extra
        assert 'stage1' in extra and 'stage2' in extra \
               and 'stage3' in extra and 'stage4' in extra
        # Assert whether the length of `num_blocks` and `num_channels` are
        # equal to `num_branches`
        for i in range(4):
            cfg = extra[f'stage{i + 1}']
            assert len(cfg['num_blocks']) == cfg['num_branches'] and \
                   len(cfg['num_channels']) == cfg['num_branches']

        cfg = self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.transformer_norm_cfg = transformer_norm_cfg
        self.frozen_stages = frozen_stages

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn1 = build_norm_layer(self.norm_cfg, 64)[1]
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn2 = build_norm_layer(self.norm_cfg, 64)[1]
        self.relu = nn.ReLU(inplace=True)

        # stochastic depth
        depths = [
            cfg[stage]['num_blocks'][0] * cfg[stage]['num_modules']
            for stage in ['stage2', 'stage3', 'stage4']
        ]
        depth_s2, depth_s3, _ = depths
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        self.stage1_cfg = cfg['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block = self.blocks_dict[self.stage1_cfg['block']]
        num_blocks = self.stage1_cfg['num_blocks'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block = self.blocks_dict[self.stage2_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channel],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, drop_path=dpr[0:depth_s2])

        self.stage3_cfg = cfg['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block = self.blocks_dict[self.stage3_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
            drop_path=dpr[depth_s2:depth_s2 + depth_s3])

        self.stage4_cfg = cfg['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block = self.blocks_dict[self.stage4_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multiscale_output=multiscale_output,
            drop_path=dpr[depth_s2 + depth_s3:])

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, (nn.LayerNorm, _BatchNorm)):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, True)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i] if j == i -
                        num_branches_pre else inchannels)
                    conv3x3s.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                inchannels,
                                outchannels,
                                3,
                                2,
                                1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, outchannels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self,
        block,
        inplanes,
        planes,
        blocks,
        num_heads=1,
        stride=1,
        window_size=7,
        mlp_ratio=4.0,
    ):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1],
            )
        layers = []

        if isinstance(block, HRFormerBlock):
            layers.append(
                block(
                    inplanes,
                    planes,
                    num_heads,
                    window_size,
                    mlp_ratio,
                    init_cfg=None))
        else:
            layers.append(
                block(
                    inplanes,
                    planes,
                    stride=stride,
                    downsample=downsample,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    init_cfg=None))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self,
                    layer_config,
                    num_inchannels,
                    multiscale_output=True,
                    drop_path=0.0):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]
        num_heads = layer_config['num_heads']
        num_window_sizes = layer_config['window_sizes']
        num_mlp_ratios = layer_config['mlp_ratios']

        modules = []
        for i in range(num_modules):
            # multiscale_output is only used last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                HRFomerModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    reset_multiscale_output,
                    drop_path=drop_path[num_blocks[0] * i:num_blocks[0] *
                                        (i + 1)],
                    norm_cfg=self.norm_cfg,
                    transformer_norm_cfg=self.transformer_norm_cfg,
                    conv_cfg=self.conv_cfg,
                    init_cfg=None))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list
