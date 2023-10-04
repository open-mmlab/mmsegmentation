# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.registry import init_default_scope

from mmseg.models.backbones.uniformer_light import (CBlock, PatchEmbed,
                                                    UniFormer_Light,
                                                    head_embedding)
from .utils import check_norm_state

init_default_scope('mmseg')


def test_uniformer_light_backbone():
    model = UniFormer_Light(
        depth=[3, 5, 9, 3],
        conv_stem=True,
        prune_ratio=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5]],
        trade_off=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5]],
        embed_dim=[64, 128, 256, 512],
        head_dim=32,
        mlp_ratio=[3, 3, 3, 3],
        drop_path_rate=0.1)
    temp = torch.randn(1, 3, 512, 512)
    model.train()
    feats = model(temp)
    assert check_norm_state(model.modules(), True)
    assert feats[-1].shape == (1, 512, 16, 16)


def test_head_embedding():
    temp = torch.randn(1, 3, 512, 512)
    head_embed = head_embedding(in_channels=3, out_channels=64)
    output = head_embed(temp)
    assert output.shape == (1, 64, 128, 128)


def test_patch_embed_block():
    temp = torch.randn(1, 64, 128, 128)
    patch_embed2 = PatchEmbed(patch_size=2, in_chans=64, embed_dim=128)
    output = patch_embed2(temp)
    assert output.shape == (1, 128, 64, 64)


def test_cblock():
    temp = torch.randn(1, 64, 128, 128)
    cblock = CBlock(
        dim=64,
        num_heads=2,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0,
        attn_drop=0.,
        drop_path=0.025,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm)
    output = cblock(temp)
    assert output.shape == temp.shape
