# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES


class PatchTransform(BaseModule):
    """ Patch Transform Layer

    Args:
        in_channels (int): The num of input channels.
        out_channels (int): The num of output channels.
        patch_size (int): The patch size.
        bias (bool): Bias of embed conv. Default: True.
        drop_rate (float, optional): Dropout rate. Default: 0.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_size,    
                 bias=True,
                 drop_rate=0., 
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=True)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        else:
            self.norm = None
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, x, hw_shape):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in) or (B, C_in, H, W)
            hw_shape (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.
        """
        assert len(x.shape) in [3, 4]
        
        H, W = hw_shape
        # reshape
        if len(x.shape) == 3:
            B, N, C = x.shape
            x = x.view(B,H,W,C).permute(0,3,1,2).contiguous()   # (B, N, C) -> (B, C, H, W)
        else:
            B, C, _, _ = x.shape
        # transform
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()   # (B, N, C)
        x = self.norm(x) if self.norm else x
        x = self.drop(x) 
        new_H, new_W = H // self.patch_size, W // self.patch_size

        return x, (new_H, new_W)


class CPE(nn.Module):
    """ Conditional Position Encoding (CPE) Module

    Args:
        dim (int): The channel number of input feature.
        kernel_size (int): The kernel size of convolution operation. 
            Default: 3.
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                              stride=1, padding=kernel_size // 2, groups=dim, bias=True)

    def forward(self, x, hw_shape):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in) or (B, C_in, H, W)
            hw_shape (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.
        """
        B, N, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, N, C) -> (B, C, H, W)
        x = self.conv(x)
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()  # (B, C, H, W) -> (B, N, C)
        return x


class PGMSA(BaseModule):
    """Pyramid Group Multi-Self Attention (PG-MSA) module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        group_num (int): The group number. 
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 group_num,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.group_num = group_num  
        self.group_axis = int(group_num ** 0.5)
        self.num_heads = num_heads
        self.head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_embed_dims**-0.5
        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.kv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.drop = build_dropout(dropout_layer)

    def split_group(self, x, B, D, H, W, _H, _W):
        '''
        input:  (B,head_num,N,D)
        output: (B,group_num,head_num,_H*_W,D)
        Args:
            x: input feature map
            B: batch size 
            D: embed_dims per head
            H: height of input feature 
            W: width of input feature 
            _H: height of each group
            _W: width of each group
        '''
        # 1. (B,head_num,N,D) -> (B*head_num,H,W,D) -> (B*head_num,D,H,W)
        x = x.view(B * self.num_heads, H, W, D).permute(0, 3, 1, 2).contiguous()
        # 2. unfold
        x = torch.nn.functional.unfold(x, kernel_size=(_H, _W), stride=(_H, _W))  # (B*head_num, D*_H*_W, R^2)
        # 3. reshape & permute
        x = x.view(B, self.num_heads, D, _H * _W, self.group_num).permute(0, 1, 4, 3,
                                                                          2).contiguous() # (B, head_num, R^2, _H*_W, D)
        return x

    def forward(self, x, hw_shape):
        """
        Args:
            x (tensor): input features with shape of (B, N, C)
            hw_shape (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.
        """
        B, N, C = x.shape
        H, W = hw_shape
        assert N == H * W
        if self.group_num == 1:  # do not split group
            # --- 1. qkv projection ---
            # q: (B, N, C) -> (B, N, head_num, D) -> (B, head_num, N, D)
            q = self.q(x).view(B, N, self.num_heads, self.head_embed_dims).permute(0, 2, 1, 3).contiguous()  # (B, head_num, N, D)
            # kv: (B, N, 2C) -> (B, N, 2, head_num, D) -> (2, B, head_num, N, D)
            kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_embed_dims).permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1]      # (B, head_num, N, D)
            # --- 2. attn ---
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else: 
            # --- 1. check if need padding ---
            pad_opt = False
            if H % self.group_axis != 0 or W % self.group_axis != 0:
                pad_opt = True
                x = x.reshape(B, H, W, C)  # (B, N, C) -> (B, H, W, C)
                # padding right & below 
                pad_right = self.group_axis - W % self.group_axis
                pad_bottom = self.group_axis - H % self.group_axis
                x = F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
                H += pad_bottom
                W += pad_right
                N = H * W  
                x = x.reshape(B, N, C)    # (B, H, W, C) -> (B, N, C)
            
            # --- 2. qkv projection ---
            _H, _W = H // self.group_axis, W // self.group_axis  # height and width of each group 
            # q: (B, N, C) -> (B, N, head_num, D) -> (B, head_num, N, D)
            q = self.q(x).view(B, N, self.num_heads, self.head_embed_dims).permute(0, 2, 1, 3).contiguous()  # (B, head_num, N, D)
            # kv: (B, N, 2C) -> (B, N, 2, head_num, D) -> (2, B, head_num, N, D)
            kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_embed_dims).permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1]  # (B, head_num, N, D)

            # --- 3. split group ---
            q, k, v = self.split_group(q, B, self.head_embed_dims, H, W, _H, _W), \
                    self.split_group(k, B, self.head_embed_dims, H, W, _H, _W), \
                    self.split_group(v, B, self.head_embed_dims, H, W, _H, _W)  # (B, head_num, R^2, _H*_W, D)

            # --- 4. attn inside each group ---
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, head_num, R^2, _H*_W, _H*_W)
            attn = attn.softmax(dim=-1)  # (B, head_num, R^2, _H*_W, _H*_W)
            attn = self.attn_drop(attn)  # (B, head_num, R^2, _H*_W, _H*_W)
            x = (attn @ v)  # (B, head_num, R^2, _H*_W, D)

            # --- 5. reorganize ---
            # (B, head_num, R^2, _H*_W, D) -> (B, head_num, D, _H*_W, R^2) -> (B*head_num, D*_H*_W, R^2)
            x = x.permute(0, 1, 4, 3, 2).reshape(B * self.num_heads, self.head_embed_dims * _H * _W, self.group_num).contiguous()  # (B*head_num, D*_H*_W, R^2)
            # fold
            x = torch.nn.functional.fold(x, output_size=(H, W), kernel_size=(_H,_W), stride=(_H,_W))  # (B*head_num, D, H, W)
            # (B*head_num, D, H, W) - > (B,N,C)
            x = x.view(B, self.num_heads, self.head_embed_dims, N).view(B, self.num_heads * self.head_embed_dims, N).permute(0, 2, 1).contiguous()  # (B,N,C)

            # --- 6. if need remove padding --- 
            if pad_opt:
                x = x.reshape(B, H, W, C)   # reshape (B, N, C) -> (B, H, W, C)
                x = x[:, :(H-pad_bottom), :(W-pad_right), :].contiguous()  # remove
                x = x.reshape(B, -1, C)     # reshape (B, H, W, C) -> (B, N, C)

        # --- project ---
        x = self.proj(x)  # (B, N, C)
        x = self.proj_drop(x)  # (B, N, C)
        # --- drop path ---
        x = self.drop(x)
        return x
            
    
class PGTBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        group_num (int): The group number. 
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 group_num,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(PGTBlock, self).__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        # attn
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = PGMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            group_num=group_num,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)
        # ffn
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
        # cpe
        self.cpe = CPE(dim=embed_dims)

    def forward(self, x, hw_shape):
        """
        Args:
            x (tensor): input features with shape of (B, N, C)
            hw_shape (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.
        """
        def _inner_forward(x, hw_shape):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            pos = self.cpe(x, hw_shape)
            x = x + pos

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, hw_shape)
        else:
            x = _inner_forward(x, hw_shape)

        return x


class PGTStage(BaseModule):
    """Implements one stage in Pyramid Group Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        group_num (int): The group number. 
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        transform (BaseModule | None, optional): The patch transform layer.
            Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 group_num,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 transform=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = PGTBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                group_num=group_num,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.transform = transform

    def forward(self, x, hw_shape):
        """
        Args:
            x (tensor): input features 
            hw_shape (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.
        """
        # patch transform
        if self.transform:
            x, hw_shape = self.transform(x, hw_shape)

        # blocks
        for block in self.blocks:
            x = block(x, hw_shape)
        
        return x, hw_shape


@BACKBONES.register_module()
class PyramidGroupTransformer(BaseModule):
    """Pyramid Group Transformer backbone.

    This backbone is the implementation of the encoder in 
    `Fully Transformer Networks for Semantic Image Segmentation <https://arxiv.org/abs/2106.04108>`_.
    
    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 64.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        group_nums (tuple[int]): Group numbers of each Swin Transformer stage. 
            Default: (64, 16, 1, 1).
        mlp_ratio (int | float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (2, 4, 8, 16).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=64,
                 patch_size=4,
                 group_nums=(64, 16, 1, 1),
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(2, 4, 8, 16),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None):
        self.frozen_stages = frozen_stages

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        
        super(PyramidGroupTransformer, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        
        # dims of each stage
        self.channels = [int(embed_dims * 2**i) for i in range(num_layers)]
        
        # stages
        self.stages = ModuleList()
        for i in range(num_layers):
            # patch transform
            transform = PatchTransform(
                in_channels=in_channels if i == 0 else self.channels[i-1],
                out_channels=self.channels[i],
                patch_size=strides[i],
                drop_rate=drop_rate if i == 0 else 0.,
                norm_cfg=norm_cfg if patch_norm else None,
                init_cfg=None)
            # blocks
            stage = PGTStage(
                embed_dims=self.channels[i],
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * self.channels[i]),
                depth=depths[i],
                group_num=group_nums[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                transform=transform,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)  
            self.stages.append(stage)
        
        # Add a norm layer for each output 
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.channels[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
        
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(PyramidGroupTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

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
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

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
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        hw_shape = x.shape[2:]

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape, self.channels[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs
    

