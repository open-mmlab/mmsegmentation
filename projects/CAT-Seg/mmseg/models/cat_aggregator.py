# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.model import BaseModule
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS
from ..utils import FullAttention, LinearAttention


class AGWindowMSA(BaseModule):
    """Appearance Guidance Window based multi-head self-attention (W-MSA)
    module with relative position bias.

    Args:
        embed_dims (int): Number of input channels.
        appearance_dims (int): Number of appearance guidance feature channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 appearance_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.appearance_dims = appearance_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qk = nn.Linear(
            embed_dims + appearance_dims, embed_dims * 2, bias=qkv_bias)
        self.v = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C),
                C = embed_dims + appearance_dims.
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, _ = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads,
                                self.embed_dims // self.num_heads).permute(
                                    2, 0, 3, 1,
                                    4)  # 2 B NUM_HEADS N embed_dims//NUM_HEADS
        v = self.v(x[:, :, :self.embed_dims]).reshape(
            B, N, self.num_heads, self.embed_dims // self.num_heads).permute(
                0, 2, 1, 3)  # B NUM_HEADS N embed_dims//NUM_HEADS
        # make torchscript happy (cannot use tensor as tuple)
        q, k = qk[0], qk[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        """Double step sequence."""
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class AGShiftWindowMSA(BaseModule):
    """Appearance Guidance Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        appearance_dims (int): Number of appearance guidance channels
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 appearance_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = AGWindowMSA(
            embed_dims=embed_dims,
            appearance_dims=appearance_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        """
        Args:
            query: The input query.
            hw_shape: The shape of the feature height and width.
        """
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size,
                                         self.w_msa.embed_dims)

        # B H' W' self.w_msa.embed_dims
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, self.w_msa.embed_dims)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class AGSwinBlock(BaseModule):
    """Appearance Guidance Swin Transformer Block.

    Args:
        embed_dims (int): The feature dimension.
        appearance_dims (int): The appearance guidance dimension.
        num_heads (int): Parallel attention heads.
        mlp_ratios (int): The hidden dimension ratio w.r.t. embed_dims
            for FFNs.
        window_size (int, optional): The local window scale.
            Default: 7.
        shift (bool, optional): whether to shift window or not.
            Default False.
        qkv_bias (bool, optional): enable bias for qkv if True.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate.
            Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate.
            Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 appearance_dims,
                 num_heads,
                 mlp_ratios=4,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = AGShiftWindowMSA(
            embed_dims=embed_dims,
            appearance_dims=appearance_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * mlp_ratios,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, inputs, hw_shape):
        """
        Args:
            inputs (list[Tensor]): appearance_guidance (B, H, W, C);
                x (B, L, C)
            hw_shape (tuple[int]): shape of feature.
        """
        x, appearance_guidance = inputs
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'

        identity = x
        x = self.norm1(x)

        # appearance guidance
        x = x.view(B, H, W, C)
        if appearance_guidance is not None:
            x = torch.cat([x, appearance_guidance], dim=-1).flatten(1, 2)

        x = self.attn(x, hw_shape)

        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)

        return x


@MODELS.register_module()
class SpatialAggregateLayer(BaseModule):
    """Spatial aggregation layer of CAT-Seg.

    Args:
        embed_dims (int): The feature dimension.
        appearance_dims (int): The appearance guidance dimension.
        num_heads (int): Parallel attention heads.
        mlp_ratios (int): The hidden dimension ratio w.r.t. embed_dims
            for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 appearance_dims,
                 num_heads,
                 mlp_ratios,
                 window_size=7,
                 qk_scale=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.block_1 = AGSwinBlock(
            embed_dims,
            appearance_dims,
            num_heads,
            mlp_ratios,
            window_size=window_size,
            shift=False,
            qk_scale=qk_scale)
        self.block_2 = AGSwinBlock(
            embed_dims,
            appearance_dims,
            num_heads,
            mlp_ratios,
            window_size=window_size,
            shift=True,
            qk_scale=qk_scale)
        self.guidance_norm = nn.LayerNorm(
            appearance_dims) if appearance_dims > 0 else None

    def forward(self, x, appearance_guidance):
        """
        Args:
            x (torch.Tensor): B C T H W.
            appearance_guidance (torch.Tensor): B C H W.
        """
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1, 2)  # BT, HW, C
        if appearance_guidance is not None:
            appearance_guidance = appearance_guidance.repeat(
                T, 1, 1, 1).permute(0, 2, 3, 1)  # BT, HW, C
            appearance_guidance = self.guidance_norm(appearance_guidance)
        else:
            assert self.appearance_dims == 0
        x = self.block_1((x, appearance_guidance), (H, W))
        x = self.block_2((x, appearance_guidance), (H, W))
        x = x.transpose(1, 2).reshape(B, T, C, -1)
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x


class AttentionLayer(nn.Module):
    """Attention layer for ClassAggregration of CAT-Seg.

    Source: https://github.com/KU-CVLAB/CAT-Seg/blob/main/cat_seg/modeling/transformer/model.py#L310 # noqa
    """

    def __init__(self,
                 hidden_dim,
                 guidance_dim,
                 nheads=8,
                 attention_type='linear'):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        if attention_type == 'linear':
            self.attention = LinearAttention()
        elif attention_type == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError

    def forward(self, x, guidance=None):
        """
        Args:
            x: B*H_p*W_p, T, C
            guidance: B*H_p*W_p, T, C
        """
        B, L, _ = x.shape
        q = self.q(torch.cat([x, guidance],
                             dim=-1)) if guidance is not None else self.q(x)
        k = self.k(torch.cat([x, guidance],
                             dim=-1)) if guidance is not None else self.k(x)
        v = self.v(x)

        q = q.reshape(B, L, self.nheads, -1)
        k = k.reshape(B, L, self.nheads, -1)
        v = v.reshape(B, L, self.nheads, -1)

        out = self.attention(q, k, v)
        out = out.reshape(B, L, -1)
        return out


@MODELS.register_module()
class ClassAggregateLayer(BaseModule):
    """Class aggregation layer of CAT-Seg.

    Args:
        hidden_dims (int): The feature dimension.
        guidance_dims (int): The appearance guidance dimension.
        num_heads (int): Parallel attention heads.
        attention_type (str): Type of attention layer. Default: 'linear'.
        pooling_size (tuple[int] | list[int]): Pooling size.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(
            self,
            hidden_dims=64,
            guidance_dims=64,
            num_heads=8,
            attention_type='linear',
            pooling_size=(4, 4),
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.pool = nn.AvgPool2d(pooling_size)
        self.attention = AttentionLayer(
            hidden_dims,
            guidance_dims,
            nheads=num_heads,
            attention_type=attention_type)
        self.MLP = FFN(
            embed_dims=hidden_dims,
            feedforward_channels=hidden_dims * 4,
            num_fcs=2)
        self.norm1 = nn.LayerNorm(hidden_dims)
        self.norm2 = nn.LayerNorm(hidden_dims)

    def pool_features(self, x):
        """Intermediate pooling layer for computational efficiency.

        Args:
            x: B, C, T, H, W
        """
        B, C, T, H, W = x.shape
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        x = self.pool(x)
        *_, H_, W_ = x.shape
        x = x.reshape(B, T, C, H_, W_).transpose(1, 2)
        return x

    def forward(self, x, guidance):
        """
        Args:
            x: B, C, T, H, W
            guidance: B, T, C
        """
        B, C, T, H, W = x.size()
        x_pool = self.pool_features(x)
        *_, H_pool, W_pool = x_pool.size()

        x_pool = x_pool.permute(0, 3, 4, 2, 1).reshape(-1, T, C)
        # B*H_p*W_p T C
        if guidance is not None:
            guidance = guidance.repeat(H_pool * W_pool, 1, 1)

        x_pool = x_pool + self.attention(self.norm1(x_pool),
                                         guidance)  # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool))  # MLP

        x_pool = x_pool.reshape(B, H_pool * W_pool, T,
                                C).permute(0, 2, 3, 1).reshape(
                                    B, T, C, H_pool,
                                    W_pool).flatten(0, 1)  # BT C H_p W_p
        x_pool = F.interpolate(
            x_pool, size=(H, W), mode='bilinear', align_corners=True)
        x_pool = x_pool.reshape(B, T, C, H, W).transpose(1, 2)  # B C T H W
        x = x + x_pool  # Residual

        return x


@MODELS.register_module()
class AggregatorLayer(BaseModule):
    """Single Aggregator Layer of CAT-Seg."""

    def __init__(self,
                 embed_dims=64,
                 text_guidance_dims=512,
                 appearance_guidance_dims=512,
                 num_heads=4,
                 mlp_ratios=4,
                 window_size=7,
                 attention_type='linear',
                 pooling_size=(2, 2),
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.spatial_agg = SpatialAggregateLayer(
            embed_dims,
            appearance_guidance_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            window_size=window_size)
        self.class_agg = ClassAggregateLayer(
            embed_dims,
            text_guidance_dims,
            num_heads=num_heads,
            attention_type=attention_type,
            pooling_size=pooling_size)

    def forward(self, x, appearance_guidance, text_guidance):
        """
        Args:
            x: B C T H W
        """
        x = self.spatial_agg(x, appearance_guidance)
        x = self.class_agg(x, text_guidance)
        return x


@MODELS.register_module()
class CATSegAggregator(BaseModule):
    """CATSeg Aggregator.

    This Aggregator is the mmseg implementation of
    `CAT-Seg <https://arxiv.org/abs/2303.11797>`_.

    Args:
        text_guidance_dim (int): Text guidance dimensions. Default: 512.
        text_guidance_proj_dim (int): Text guidance projection dimensions.
            Default: 128.
        appearance_guidance_dim (int): Appearance guidance dimensions.
            Default: 512.
        appearance_guidance_proj_dim (int): Appearance guidance projection
            dimensions. Default: 128.
        num_layers (int): Aggregator layer number. Default: 4.
        num_heads (int): Attention layer head number. Default: 4.
        embed_dims (int): Input feature dimensions. Default: 128.
        pooling_size (tuple | list): Pooling size of the class aggregator
            layer. Default: (6, 6).
        mlp_ratios (int): The hidden dimension ratio w.r.t. input dimension.
            Default: 4.
        window_size (int): Swin block window size. Default:12.
        attention_type (str): Attention type of class aggregator layer.
            Default:'linear'.
        prompt_channel (int): Prompt channels. Default: 80.
    """

    def __init__(self,
                 text_guidance_dim=512,
                 text_guidance_proj_dim=128,
                 appearance_guidance_dim=512,
                 appearance_guidance_proj_dim=128,
                 num_layers=4,
                 num_heads=4,
                 embed_dims=128,
                 pooling_size=(6, 6),
                 mlp_ratios=4,
                 window_size=12,
                 attention_type='linear',
                 prompt_channel=80,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dims = embed_dims

        self.layers = nn.ModuleList([
            AggregatorLayer(
                embed_dims=embed_dims,
                text_guidance_dims=text_guidance_proj_dim,
                appearance_guidance_dims=appearance_guidance_proj_dim,
                num_heads=num_heads,
                mlp_ratios=mlp_ratios,
                window_size=window_size,
                attention_type=attention_type,
                pooling_size=pooling_size) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(
            prompt_channel, embed_dims, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(
                appearance_guidance_dim,
                appearance_guidance_proj_dim,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

    def feature_map(self, img_feats, text_feats):
        """Concatenation type cost volume.

        For ablation study of cost volume type.
        """
        img_feats = F.normalize(img_feats, dim=1)  # B C H W
        img_feats = img_feats.unsqueeze(2).repeat(1, 1, text_feats.shape[1], 1,
                                                  1)
        text_feats = F.normalize(text_feats, dim=-1)  # B T P C
        text_feats = text_feats.mean(dim=-2)
        text_feats = F.normalize(text_feats, dim=-1)  # B T C
        text_feats = text_feats.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, 1, img_feats.shape[-2], img_feats.shape[-1]).transpose(1, 2)
        return torch.cat((img_feats, text_feats), dim=1)  # B 2C T H W

    def correlation(self, img_feats, text_feats):
        """Correlation of image features and text features."""
        img_feats = F.normalize(img_feats, dim=1)  # B C H W
        text_feats = F.normalize(text_feats, dim=-1)  # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        """Correlation embeddings encoding."""
        B = x.shape[0]
        corr_embed = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        corr_embed = self.conv1(corr_embed)
        corr_embed = corr_embed.reshape(B, -1, self.embed_dims, x.shape[-2],
                                        x.shape[-1]).transpose(1, 2)
        return corr_embed

    def forward(self, inputs):
        """
        Args:
            inputs (dict): including the following keys,
                'appearance_feat': list[torch.Tensor], w.r.t. out_indices of
                    `self.feature_extractor`.
                'clip_text_feat': the text feature extracted by clip text
                    encoder.
                'clip_text_feat_test': the text feature extracted by clip text
                    encoder for testing.
                'clip_img_feat': the image feature extracted clip image
                    encoder.
        """
        img_feats = inputs['clip_img_feat']
        B = img_feats.size(0)
        appearance_guidance = inputs[
            'appearance_feat'][::-1]  # order (out_indices) 2, 1, 0
        text_feats = inputs['clip_text_feat'] if self.training else inputs[
            'clip_text_feat_test']
        text_feats = text_feats.repeat(B, 1, 1, 1)

        corr = self.correlation(img_feats, text_feats)
        # corr = self.feature_map(img_feats, text_feats)
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance = None, None

        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(
                appearance_guidance[0])

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance,
                               projected_text_guidance)

        return dict(
            corr_embed=corr_embed, appearance_feats=appearance_guidance[1:])
