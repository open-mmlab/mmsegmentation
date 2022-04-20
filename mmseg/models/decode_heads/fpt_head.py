# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import ModuleList
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)


@HEADS.register_module()
class FeaturePyramidTransformerHead(BaseDecodeHead):
    """Fully Transformer Networks for Semantic Image Segmentation.

    This head is the implementation of
    `<https://arxiv.org/abs/2106.04108>`_.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        num_layers (tuple[int]): The layer numbers.
            Default: ((1, 1, 1), (1, 1), (1)).  
        num_heads (int): The number of attention heads.
            Default: (4, 4, 4).
        sra_ratios (tuple[int]): The layer numbers.
            Default: ((2, 2, 2), (2, 2), (2)).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (tuple[float]): stochastic depth rate of each branch. 
            Default (0.3, 0.2, 0.1).
        use_ape (bool): Use absolute position encoding for each branch if True. Default: False.
        ape_sizes (tuple[int]): size of absolute position encoding. Default: [(16, 16), (32, 32), (64, 64), (128, 128)].
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        init_std (float): The value of std in weight initialization.
            Default: 0.02.
    """

    def __init__(
            self,
            num_layers=((1, 1, 1), (1, 1), (1)),
            num_heads=4,
            sra_ratios=((2, 2, 2), (2, 2), (2)), 
            mlp_ratio=4,
            num_fcs=2,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=(0.3, 0.2, 0.1),
            use_ape=False,
            ape_sizes=[(16, 16), (32, 32), (64, 64), (128, 128)],
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            init_std=0.02,
            **kwargs,
    ):
        super(FeaturePyramidTransformerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.branches = ModuleList()
        for branch_idx, block_nums in enumerate(num_layers):  # block_nums=(1,1,1)
            branch = ModuleList()
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate[branch_idx], sum(block_nums))]  
            cur = 0 
            for stage_idx, block_num in enumerate(block_nums):  # block_num: 1
                stage = ModuleList( 
                    FPTBlock(
                    embed_dims=self.channels,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * self.channels,
                    sra_ratios=sra_ratios[branch_idx][stage_idx],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + i],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg)
                    for i in range(block_num))
                cur = cur + block_num
                # branch.append(nn.Sequential(*stage))
                branch.append(stage)
            self.branches.append(branch)

        # APE
        self.use_ape = use_ape
        if use_ape: 
            self.ape_sizes = ape_sizes
            self.ape_embeds = nn.ParameterList()
            self.ape_drops = ModuleList()
            for branch_idx in range(len(num_layers)):
                ape_embed = nn.Parameter(torch.zeros(1, self.channels, ape_sizes[branch_idx][0], ape_sizes[branch_idx][1])) 
                ape_drop = nn.Dropout2d(p=drop_rate)
                self.ape_embeds.append(ape_embed)
                self.ape_drops.append(ape_drop)

        self.init_std = init_std

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
        # init ape
        if self.use_ape:
            for i in range(len(self.ape_embeds)):
                trunc_normal_(self.ape_embeds[i], std=self.init_std) 

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)  # list
        B, C, _, _  = inputs[-1].shape

        outs = []
        for branch_idx, branch in enumerate(self.branches):
            x = inputs[-1 - branch_idx]
            H, W = x.shape[2:]
            N = H * W
            # --- if ape ---
            if self.use_ape:
                pos = self.ape_embeds[branch_idx]  # (1,C,H_ape,W_ape)
                if (H, W) != self.ape_sizes[branch_idx]:  # need interpolation
                    pos = F.interpolate(pos, size=(H, W), mode='bilinear', align_corners=self.align_corners)  # (1,C,H_ape,W_ape) -> (1,C,H,W)
                x = x + pos  # (B,C,H,W)
                x = self.ape_drops[branch_idx](x)  

            # --- stages in each branch ---
            for stage_idx, stage in enumerate(branch):
                x = x.view(B, C, N).permute(0, 2, 1).contiguous()  # (B,C,H,W) -> (B,N,C)
                # blocks
                for block in stage: 
                    x = block(x, hw_shape=(H, W))  
                # upsample
                if stage_idx < (len(branch) - 1):
                    x = x.view(B, H, W, C).permute(0, 3, 2, 1).contiguous()  # (B,N,C) -> (B,C,H,W)
                    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=self.align_corners)  # (B,C,2H,2W)
                    H, W = x.shape[2:]
                    N = H * W
            x = x.view(B, H, W, C).permute(0, 3, 2, 1).contiguous()  # (B,C,H,W)
            outs.append(x)

        # add
        for i, x in enumerate(outs):
            if i == 0:
                out = x
            else:
                out = out + x
        # seg head
        out = self.cls_seg(out)  # (B, K, H, W)
        
        return out
                

class FPTBlock(nn.Module):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        sra_ratios (int): The spatial reduction ratio. 
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
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
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 sra_ratios,
                 num_fcs=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN')):
        
        super(FPTBlock, self).__init__()

        # attn
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = SRMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            sra_ratios=sra_ratios,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))
        # ffn
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)

        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)

        return x


class SRMSA(nn.Module):
    """Spatial Reduction Multi-Self Attention (SR-MSA) module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        sra_ratios (int): The spatial reduction ratio. 
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 sra_ratios,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.)):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_embed_dims**-0.5

        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.kv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.drop = build_dropout(dropout_layer)

        self.sra_ratios = sra_ratios  
        if sra_ratios > 1:
            self.sr = nn.Conv2d(embed_dims, embed_dims, kernel_size=sra_ratios, stride=sra_ratios)
            self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x, hw_shape):
        """
        Args:
            x (tensor): input features with shape of (B, N, C)
            hw_shape (tensor,int): shape (H,W)
        """
        B, N, C = x.shape
        H, W = hw_shape
        assert N == H * W
        # q projection
        q = self.q(x).view(B, N, self.num_heads, self.head_embed_dims).permute(0, 2, 1, 3).contiguous()   # (B, head, N, D)
        # kv projection
        if self.sra_ratios > 1:
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            x = self.sr(x).view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, C, H_down, W_down) -> (B, N_down, C)
            x = self.norm(x)  # (B, N_down, C)
        kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_embed_dims).permute(2, 0, 3, 1, 4).contiguous()   # (2, B, head, N_down, D) or (2, B, head, N, D)
        k, v = kv[0], kv[1]    # (B, head, N_down, D) or (B, head, N, D)
        # attn
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, head, N, N_down) or (B, head, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        # project
        x = self.proj(x)
        x = self.proj_drop(x)
        # drop path
        x = self.drop(x) 

        return x




