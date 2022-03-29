# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed

try:
    from scipy import interpolate
except ImportError:
    interpolate = None


class BEiTAttention(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qv_bias (bool):  If True, add a learnable bias to q, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        if qv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.v_bias = None

        self.window_size = window_size
        # cls to token & token 2 cls & cls to cls
        self.num_relative_distance = (2 * window_size[0] -
                                      1) * (2 * window_size[1] - 1) + 3
        # relative_position_bias_table shape is (2*Wh-1 * 2*Ww-1 + 3, nH)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))

        # get pair-wise relative position index for
        # each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        # coords shape is (2, Wh, Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # coords_flatten shape is (2, Wh*Ww)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :])
        # relative_coords shape is (Wh*Ww, Wh*Ww, 2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1, ) * 2,
            dtype=relative_coords.dtype)
        # relative_position_index shape is (Wh*Ww, Wh*Ww)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer('relative_position_index',
                             relative_position_index)
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C).
        """
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.relative_position_bias_table is not None:
            Wh = self.window_size[0]
            Ww = self.window_size[1]
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    Wh * Ww + 1, Wh * Ww + 1, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qv_bias (bool): Enable bias for qv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        window_size (tuple[int], optional): The height and width of the window.
            Default: None.
        init_values (float, optional): Initialize the values of BEiTAttention
            and FFN with learnable scaling. Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 window_size=None,
                 init_values=None):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.attn = BEiTAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            qv_bias=qv_bias,
            qk_scale=None,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=0.,
            init_cfg=None)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=0.,
            dropout_layer=None,
            act_cfg=act_cfg,
            add_identity=False)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        # NOTE: drop path for stochastic depth, we shall see if
        # this is better than dropout here
        dropout_layer = dict(type='DropPath', drop_prob=drop_path_rate)
        self.drop_path = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.gamma_1 = nn.Parameter(
            init_values * torch.ones((embed_dims)), requires_grad=True)
        self.gamma_2 = nn.Parameter(
            init_values * torch.ones((embed_dims)), requires_grad=True)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        return x


@BACKBONES.register_module()
class BEiT(BaseModule):
    """BERT Pre-Training of Image Transformers.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_layers (int): Depth of transformer. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qv_bias (bool): Enable bias for qv if True. Default: True.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        pretrained (str, optional): Model pretrained path. Default: None.
        init_values (float): Initialize the values of BEiTAttention and FFN
            with learnable scaling.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qv_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 num_fcs=2,
                 norm_eval=False,
                 pretrained=None,
                 init_values=0.1,
                 init_cfg=None):
        super(BEiT, self).__init__(init_cfg=init_cfg)
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.img_size = img_size
        self.patch_size = patch_size
        self.norm_eval = norm_eval
        self.pretrained = pretrained

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        window_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.patch_shape = window_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qv_bias=qv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    window_size=window_size,
                    init_values=init_values))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _geometric_sequence_interpolation(self, src_size, dst_size, sequence,
                                          num):
        """Get new sequence via geometric sequence interpolation.

        Args:
            src_size (int): Pos_embedding size in pre-trained model.
            dst_size (int): Pos_embedding size in the current model.
            sequence (tensor): The relative position bias of the pretrain
                model after removing the extra tokens.
            num (int): Number of attention heads.
        Returns:
            new_sequence (tensor): Geometric sequence interpolate the
                pre-trained relative position bias to the size of
                the current model.
        """

        def geometric_progression(a, r, n):
            return a * (1.0 - r**n) / (1.0 - r)

        # Here is a binary function.
        left, right = 1.01, 1.5
        while right - left > 1e-6:
            q = (left + right) / 2.0
            gp = geometric_progression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q
        # The position of each interpolated point is determined
        # by the ratio obtained by dichotomy.
        dis = []
        cur = 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q**(i + 1)
        r_ids = [-_ for _ in reversed(dis)]
        x = r_ids + [0] + dis
        y = r_ids + [0] + dis
        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)
        # Interpolation functions are being executed and called.
        new_sequence = []
        for i in range(num):
            z = sequence[:, i].view(src_size, src_size).float().numpy()
            f = interpolate.interp2d(x, y, z, kind='cubic')
            new_sequence.append(
                torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(sequence))
        new_sequence = torch.cat(new_sequence, dim=-1)
        return new_sequence

    def resize_rel_pos_embed(self, checkpoint):
        """Resize relative pos_embed weights.

        This function is modified from
        https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/mmcv_custom/checkpoint.py.  # noqa: E501
        Copyright (c) Microsoft Corporation
        Licensed under the MIT License

        Args:
            checkpoint (dict): Key and value of the pretrain model.
        Returns:
            state_dict (dict): Interpolate the relative pos_embed weights
                in the pre-train model to the current model size.
        """
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        all_keys = list(state_dict.keys())
        for key in all_keys:
            if 'relative_position_index' in key:
                state_dict.pop(key)
            # In order to keep the center of pos_bias as consistent as
            # possible after interpolation, and vice versa in the edge
            # area, the geometric sequence interpolation method is adopted.
            if 'relative_position_bias_table' in key:
                rel_pos_bias = state_dict[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = self.state_dict()[key].size()
                dst_patch_shape = self.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                # Count the number of extra tokens.
                num_extra_tokens = dst_num_pos - (
                    dst_patch_shape[0] * 2 - 1) * (
                        dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens)**0.5)
                dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
                if src_size != dst_size:
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
                    new_rel_pos_bias = self._geometric_sequence_interpolation(
                        src_size, dst_size, rel_pos_bias, num_attn_heads)
                    new_rel_pos_bias = torch.cat(
                        (new_rel_pos_bias, extra_tokens), dim=0)
                    state_dict[key] = new_rel_pos_bias

        return state_dict

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            state_dict = self.resize_rel_pos_embed(checkpoint)
            self.load_state_dict(state_dict, False)
        elif self.init_cfg is not None:
            super(BEiT, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            # Copyright 2019 Ross Wightman
            # Licensed under the Apache License, Version 2.0 (the "License")
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                # Remove class token and reshape token for decoder head
                out = x[:, 1:]
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        super(BEiT, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
