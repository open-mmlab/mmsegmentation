# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from ..builder import BACKBONES

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., sync_norm=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        if sync_norm :
            norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, hidden_features, postfix=1)
            self.norm2_name, norm2 = build_norm_layer(norm_cfg, hidden_features, postfix=2)

            self.add_module(self.norm1_name, norm1)
            self.add_module(self.norm2_name, norm2)
        else :
            self.norm1 = nn.BatchNorm1d(hidden_features)
            self.norm2 = nn.BatchNorm1d(hidden_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.dwconv(x, H, W)
        x = self.norm2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_MaxPool(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, sync_norm=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv1d(dim, dim, 1, bias=qkv_bias)
        self.k = nn.Conv1d(dim, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            if sync_norm :
                norm_cfg = dict(type='SyncBN', requires_grad=True)
                self.norm1_name, norm1 = build_norm_layer(norm_cfg, dim, postfix=1)
                self.add_module(self.norm1_name, norm1)
            else :
                self.norm1 = nn.BatchNorm1d(dim)
        self.apply(self._init_weights)
    
    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        B, C, N = x.shape
        q = self.q(x) 
        q = q.reshape(B, self.num_heads, C // self.num_heads, N) 
        q = q.permute(0, 1, 3, 2) 

        if self.sr_ratio > 1:
            x_ = x.reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1)
            x_ = self.norm1(x_)  
            k = self.k(x_).reshape(B, self.num_heads, C // self.num_heads, -1) 
        else:
            k = self.k(x).reshape(B, self.num_heads, C // self.num_heads, -1)

        v = torch.mean(x, 2, True).repeat(1, 1, self.num_heads).transpose(-2, -1)

        attn = (q @ k) * self.scale 
        attn, _ = torch.max(attn, -1)

        out = (attn.transpose(-2, -1) @ v)
        out = out.transpose(-2, -1)

        out = self.proj(out)

        return out



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm, sr_ratio=1, sync_norm=True):
        super().__init__()
        if sync_norm :
            norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, dim, postfix=1)
            self.norm2_name, norm2 = build_norm_layer(norm_cfg, dim, postfix=2)
            self.add_module(self.norm1_name, norm1)
            self.add_module(self.norm2_name, norm2)
        else :
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(dim)

        self.attn = Attention_MaxPool(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)
        
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding """
    def __init__(self, img_size=(224,224), patch_size=7, stride=4, in_chans=3, embed_dim=768, sync_norm=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size[0] // patch_size[0] * img_size[1] // patch_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.H = (img_size[0] - patch_size[0] + 2 * (patch_size[0] // 2)) / stride + 1
        self.W = (img_size[1] - patch_size[1] + 2 * (patch_size[1] // 2)) / stride + 1
        self.feat_shape = (int(self.H), int(self.W))
        self.N = int(self.feat_shape[0] * self.feat_shape[1])

        if sync_norm:
            norm_cfg = dict(type='SyncBN', requires_grad=True)
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dim, postfix=1)
            self.add_module(self.norm1_name, norm1)
        else :
            self.norm1 = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)
        
    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm1(x)
        x = x.flatten(2)
        return x, H, W


@BACKBONES.register_module()
class SimplifiedTransformer(nn.Module):
    def __init__(self, img_size=(224,224), patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], sync_norm=True 
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.num_layers = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0], sync_norm=sync_norm)
        self.patch_embed2 = OverlapPatchEmbed(img_size=(img_size[0] // 4, img_size[1] // 4), patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1], sync_norm=sync_norm)
        self.patch_embed3 = OverlapPatchEmbed(img_size=(img_size[0] // 8, img_size[1] // 8), patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2], sync_norm=sync_norm)
        self.patch_embed4 = OverlapPatchEmbed(img_size=(img_size[0] // 16, img_size[1] // 16), patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3], sync_norm=sync_norm)

        # encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], sync_norm=sync_norm)
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], sync_norm=sync_norm)
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], sync_norm=sync_norm)
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], sync_norm=sync_norm)
            for i in range(depths[3])])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'} 

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        
        B = x.shape[0]
        outs = []
        ref_feat = {'1': [], '2': [], '3': [], '4': [],}
        
        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
            ref_feat['1'].append(x)
        x = x.reshape(B, -1, H, W).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
            ref_feat['2'].append(x)
        x = x.reshape(B, -1, H, W).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
            ref_feat['3'].append(x)
        x = x.reshape(B, -1, H, W).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
            ref_feat['4'].append(x)
        x = x.reshape(B, -1, H, W).contiguous()
        outs.append(x)

        return outs, ref_feat

    def forward(self, x):
        x, ref_feat = self.forward_features(x)
        return x, ref_feat


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, C, N = x.shape
        x = x.reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2)
        return x

