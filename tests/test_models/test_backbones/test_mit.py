# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import MixVisionTransformer
from mmseg.models.backbones.mit import EfficientMultiheadAttention, MixFFN


def test_mit():
    with pytest.raises(AssertionError):
        # It's only support official style and mmcls style now.
        MixVisionTransformer(pretrain_style='timm')

    with pytest.raises(TypeError):
        # Pretrained represents pretrain url and must be str or None.
        MixVisionTransformer(pretrained=123)

    # Test normal input
    H, W = (224, 224)
    temp = torch.randn((1, 3, H, W))
    model = MixVisionTransformer(
        embed_dims=32, num_heads=[1, 2, 5, 8], out_indices=(0, 1, 2, 3))
    model.init_weights()
    outs = model(temp)
    assert outs[0].shape == (1, 32, H // 4, W // 4)
    assert outs[1].shape == (1, 64, H // 8, W // 8)
    assert outs[2].shape == (1, 160, H // 16, W // 16)
    assert outs[3].shape == (1, 256, H // 32, W // 32)

    # Test non-squared input
    H, W = (224, 320)
    temp = torch.randn((1, 3, H, W))
    outs = model(temp)
    assert outs[0].shape == (1, 32, H // 4, W // 4)
    assert outs[1].shape == (1, 64, H // 8, W // 8)
    assert outs[2].shape == (1, 160, H // 16, W // 16)
    assert outs[3].shape == (1, 256, H // 32, W // 32)

    # Test MixFFN
    FFN = MixFFN(128, 512)
    hw_shape = (32, 32)
    token_len = 32 * 32
    temp = torch.randn((1, token_len, 128))
    # Self identity
    out = FFN(temp, hw_shape)
    assert out.shape == (1, token_len, 128)
    # Out identity
    outs = FFN(temp, hw_shape, temp)
    assert out.shape == (1, token_len, 128)

    # Test EfficientMHA
    MHA = EfficientMultiheadAttention(128, 2)
    hw_shape = (32, 32)
    token_len = 32 * 32
    temp = torch.randn((1, token_len, 128))
    # Self identity
    out = MHA(temp, hw_shape)
    assert out.shape == (1, token_len, 128)
    # Out identity
    outs = MHA(temp, hw_shape, temp)
    assert out.shape == (1, token_len, 128)
