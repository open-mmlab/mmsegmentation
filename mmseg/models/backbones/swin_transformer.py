import warnings
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, trunc_normal_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.runner import _load_checkpoint
from mmcv.runner.base_module import BaseModule, ModuleList
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.utils import _pair as to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed


class PatchMerging(BaseModule):
    """Merge patch feature map.

    This layer use nn.Unfold to group feature map by kernel_size, and use norm
    and linear layer to embed grouped feature map.
    Args:
        input_resolution (tuple): The size of input patch resolution.
        in_channels (int): The num of input channels.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults: 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer.
            Defaults: None. (Default to be equal with kernel_size).
        padding (int | tuple, optional): zero padding width in the unfold
            layer. Defaults: 0.
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Defaults: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Defaults: None.
    """

    def __init__(self,
                 input_resolution,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding=0,
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg)
        H, W = input_resolution
        self.input_resolution = input_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride is None:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)
        self.sampler = nn.Unfold(kernel_size, dilation, padding, stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        H_out = (H + 2 * padding[0] - dilation[0] *
                 (kernel_size[0] - 1) - 1) // stride[0] + 1
        W_out = (W + 2 * padding[1] - dilation[1] *
                 (kernel_size[1] - 1) - 1) // stride[1] + 1
        self.output_resolution = (H_out, W_out)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        x = self.sampler(x)  # B, 4*C, H/2*W/2
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C

        x = self.norm(x) if self.norm else x
        x = self.reduction(x)

        return x


@ATTENTION.register_module()
class WindowMSA(BaseModule):
    """Window based multi-head self attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        init_cfg (dict, optional): The Config for initialization.
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
                 init_cfg=None):

        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords_h = self.double_step_seq(2 * Ww - 1, Wh, 1, Wh)
        rel_index_coords_w = self.double_step_seq(2 * Wh - 1, Ww, 1, Ww)
        rel_position_index = rel_index_coords_h + rel_index_coords_w.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        super(WindowMSA, self).init_weights()

        trunc_normal_init(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

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
        else:
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


@ATTENTION.register_module()
class ShiftWindowMSA(BaseModule):

    def __init__(self,
                 input_resolution,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 auto_pad=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        if min(input_resolution) <= window_size:
            # if window size is larger than input resolution, don't partition
            self.shift_size = shift_size = 0
            self.window_size = window_size = min(input_resolution)

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

        H, W = input_resolution
        # Handle auto padding
        self.auto_pad = auto_pad
        if self.auto_pad:
            self.pad_r = (window_size - W % window_size) % window_size
            self.pad_b = (window_size - H % window_size) % window_size
            dummy = torch.empty((1, H, W, 1))  # 1 H W 1
            dummy = F.pad(dummy, (0, 0, 0, self.pad_r, 0, self.pad_b))
            _, self.H_pad, self.W_pad, _ = dummy.shape
        else:
            H_pad, W_pad = H, W
            assert H_pad % window_size + W_pad % window_size == 0,\
                f'input_resolution({input_resolution}) is not divisible '\
                f'by window_size({window_size}). Please check feature '\
                f'map shape or set `auto_pad=True`.'
            self.H_pad, self.W_pad = H_pad, W_pad
            self.pad_r, self.pad_b = 0, 0

        if shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, self.H_pad, self.W_pad, 1))  # 1 H W 1
            h_slices = (slice(0, -window_size), slice(-window_size,
                                                      -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size,
                                                      -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, query, **kwargs):
        H, W = self.input_resolution
        B, L, C = query.shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        if self.pad_r or self.pad_b:
            query = F.pad(query, (0, 0, 0, self.pad_r, 0, self.pad_b))

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
        else:
            shifted_query = query

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, self.H_pad, self.W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if self.auto_pad and self.pad_r or self.pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):

    def __init__(self,
                 input_resolution,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 auto_pad=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__(init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            input_resolution=input_resolution,
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            auto_pad=auto_pad,
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)
        return x


class SwinBlockSequence(BaseModule):

    def __init__(self,
                 input_resolution,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 auto_pad=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        drop_path_rate = drop_path_rate if isinstance(
            drop_path_rate,
            list) else [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                input_resolution=input_resolution,
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                auto_pad=auto_pad,
                init_cfg=None)
            self.blocks.append(block)

        if downsample:
            self.downsample = PatchMerging(
                input_resolution,
                in_channels=embed_dims,
                out_channels=2 * embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                norm_cfg=norm_cfg,
                init_cfg=None)
        else:
            self.downsample = None

    def forward(self, query):
        for block in self.blocks:
            query = block(query)

        if self.downsample:
            query = self.downsample(query)
        return query


@BACKBONES.register_module()
class SwinTransformer(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        arch (str | dict): Swin Transformer architecture
            Defaults to 'T'.
        img_size (int | tuple): The size of input image.
            Defaults to 224.
        in_channels (int): The num of input channels.
            Defaults to 3.
        drop_rate (float): Dropout rate.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate.
            Defaults to 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults to False.
        auto_pad (bool): If True, auto pad feature map to fit window_size.
            Defaults to False.
        norm_cfg (dict, optional): Config dict for normalization layer at end
            of backone. Defaults to dict(type='LN')
        stage_cfg (dict, optional): Extra config dict for stages.
            Defaults to None.
        patch_cfg (dict, optional): Extra config dict for patch embedding.
            Defaults to None.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 depths=[2, 2, 6, 2],
                 window_sizes=[7, 7, 7, 7],
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=[4, 4, 4, 4],
                 conv_types=['Conv2d', 'Conv2d', 'Conv2d', 'Conv2d'],
                 kernel_sizes=[4, 2, 2, 2],
                 strides=[None, None, None, None],
                 paddings=[0, 0, 0, 0],
                 dilations=[1, 1, 1, 1],
                 out_indices=[0, 1, 2, 3],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 auto_pad=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 pretrain_style='official',
                 pretrained=None,
                 init_cfg=None):
        super(SwinTransformer, self).__init__()

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        assert pretrain_style in ['official', 'mmcls']

        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.auto_pad = auto_pad
        self.pretrain_style = pretrain_style
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type=conv_types[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=paddings[0],
            dilation=dilations[0],
            norm_cfg=norm_cfg,
            init_cfg=None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        if self.use_abs_pos_embed:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = ModuleList()
        input_resolution = patches_resolution
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = True
                kernel_size = kernel_sizes[i + 1]
                stride = strides[i + 1]
                padding = paddings[i + 1]
                dilation = dilations[i + 1]
            else:
                downsample = False
                kernel_size = None
                stride = None
                padding = None
                dilation = None

            stage = SwinBlockSequence(
                input_resolution=input_resolution,
                embed_dims=embed_dims,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio[i] * embed_dims,
                depth=depths[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                window_size=window_sizes[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[:depths[i]],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                auto_pad=auto_pad,
                init_cfg=None)
            self.stages.append(stage)

            dpr = dpr[depths[i]:]
            if downsample:
                embed_dims = stage.downsample.out_channels
                input_resolution = stage.downsample.output_resolution

        num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def init_weights(self):
        if self.pretrained is None:
            if self.use_abs_pos_embed:
                trunc_normal_init(self.absolute_pos_embed, std=0.02)
            for m in self.modules:
                if isinstance(m, Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
        elif isinstance(self.pretrained, str):
            logger = get_root_logger()
            ckpt = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')

            # OrderedDict is a subclass of dict
            if not isinstance(ckpt, dict):
                raise RuntimeError(
                    f'No state_dict found in checkpoint file {self.pretrained}'
                )
            # get state_dict from checkpoint
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2)

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
                else:
                    if L1 != L2:
                        S1 = int(L1**0.5)
                        S2 = int(L2**0.5)
                        table_pretrained_resized = F.interpolate(
                            table_pretrained.permute(1,
                                                     0).view(1, nH1, S1, S1),
                            size=(S2, S2),
                            mode='bicubic')
                        state_dict[table_key] = table_pretrained_resized.view(
                            nH2, L2).permute(1, 0)

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                outs.append(out)

        return x.transpose(1, 2)
