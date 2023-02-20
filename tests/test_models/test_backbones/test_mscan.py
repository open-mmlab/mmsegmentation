# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.backbones import MSCAN
from mmseg.models.backbones.mscan import (MSCAAttention, MSCASpatialAttention,
                                          OverlapPatchEmbed, StemConv)


def test_mscan_backbone():
    # Test MSCAN Standard Forward
    model = MSCAN(
        embed_dims=[8, 16, 32, 64],
        norm_cfg=dict(type='BN', requires_grad=True))
    model.init_weights()
    model.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 64, 128)
    feat = model(imgs)

    assert len(feat) == 4
    # output for segment Head
    assert feat[0].shape == torch.Size([batch_size, 8, 16, 32])
    assert feat[1].shape == torch.Size([batch_size, 16, 8, 16])
    assert feat[2].shape == torch.Size([batch_size, 32, 4, 8])
    assert feat[3].shape == torch.Size([batch_size, 64, 2, 4])

    # Test input with rare shape
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 95, 27)
    feat = model(imgs)
    assert len(feat) == 4


def test_mscan_overlap_patch_embed_module():
    x_overlap_patch_embed = OverlapPatchEmbed(
        norm_cfg=dict(type='BN', requires_grad=True))
    assert x_overlap_patch_embed.proj.in_channels == 3
    assert x_overlap_patch_embed.norm.weight.shape == torch.Size([768])
    x = torch.randn(2, 3, 16, 32)
    x_out, H, W = x_overlap_patch_embed(x)
    assert x_out.shape == torch.Size([2, 32, 768])


def test_mscan_spatial_attention_module():
    x_spatial_attention = MSCASpatialAttention(8)
    assert x_spatial_attention.proj_1.kernel_size == (1, 1)
    assert x_spatial_attention.proj_2.stride == (1, 1)
    x = torch.randn(2, 8, 16, 32)
    x_out = x_spatial_attention(x)
    assert x_out.shape == torch.Size([2, 8, 16, 32])


def test_mscan_attention_module():
    x_attention = MSCAAttention(8)
    assert x_attention.conv0.weight.shape[0] == 8
    assert x_attention.conv3.kernel_size == (1, 1)
    x = torch.randn(2, 8, 16, 32)
    x_out = x_attention(x)
    assert x_out.shape == torch.Size([2, 8, 16, 32])


def test_mscan_stem_module():
    x_stem = StemConv(8, 8, norm_cfg=dict(type='BN', requires_grad=True))
    assert x_stem.proj[0].weight.shape[0] == 4
    assert x_stem.proj[-1].weight.shape[0] == 8
    x = torch.randn(2, 8, 16, 32)
    x_out, H, W = x_stem(x)
    assert x_out.shape == torch.Size([2, 32, 8])
    assert (H, W) == (4, 8)
