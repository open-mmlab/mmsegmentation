import torch

from mmseg.models.backbones import MixVisionTransformer

# import pytest


def test_mit():
    H, W = (224, 224)
    temp = torch.randn((1, 3, H, W))
    model = MixVisionTransformer(
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        out_indices=(0, 1, 2, 3))
    outs = model(temp)
    assert outs[0].shape == (1, 32, H // 4, W // 4)
    assert outs[1].shape == (1, 64, H // 8, W // 8)
    assert outs[2].shape == (1, 160, H // 16, W // 16)
    assert outs[3].shape == (1, 256, H // 32, W // 32)
