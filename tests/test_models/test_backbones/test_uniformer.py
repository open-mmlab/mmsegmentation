# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.registry import init_default_scope

from mmseg.models.backbones.uniformer import CBlock, PatchEmbed, UniFormer
from .utils import check_norm_state

init_default_scope('mmseg')


def test_uniformer_backbone():
    # Test normal input
    H, W = (512, 512)
    temp = torch.randn((1, 3, H, W))
    model = UniFormer(
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        hybrid=False,
        layers=[5, 8, 20, 7],
        mlp_ratio=4.0,
        qkv_bias=True,
        use_checkpoint=False,
        windows=False)
    model.train()
    feats = model(temp)
    assert feats[-1].shape == (1, 512, 16, 16)
    assert check_norm_state(model.modules(), True)


def test_patch_embed_block():
    temp = torch.randn(1, 3, 512, 512)
    embed_dim = [64, 128, 320, 512]
    patch_embed1 = PatchEmbed(
        img_size=224, patch_size=4, in_chans=3, embed_dim=embed_dim[0])
    output = patch_embed1(temp)
    assert output.shape == (1, 64, 128, 128)


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
