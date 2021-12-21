# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import MixVisionTransformer
from mmseg.models.backbones.mit import EfficientMultiheadAttention, MixFFN


def test_mit():
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
    H, W = (224, 256)
    temp = torch.randn((1, 3, H, W))
    outs = model(temp)
    assert outs[0].shape == (1, 32, H // 4, W // 4)
    assert outs[1].shape == (1, 64, H // 8, W // 8)
    assert outs[2].shape == (1, 160, H // 16, W // 16)
    assert outs[3].shape == (1, 256, H // 32, W // 32)

    # Test MixFFN
    FFN = MixFFN(64, 128)
    hw_shape = (32, 32)
    token_len = 32 * 32
    temp = torch.randn((1, token_len, 64))
    # Self identity
    out = FFN(temp, hw_shape)
    assert out.shape == (1, token_len, 64)
    # Out identity
    outs = FFN(temp, hw_shape, temp)
    assert out.shape == (1, token_len, 64)

    # Test EfficientMHA
    MHA = EfficientMultiheadAttention(64, 2)
    hw_shape = (32, 32)
    token_len = 32 * 32
    temp = torch.randn((1, token_len, 64))
    # Self identity
    out = MHA(temp, hw_shape)
    assert out.shape == (1, token_len, 64)
    # Out identity
    outs = MHA(temp, hw_shape, temp)
    assert out.shape == (1, token_len, 64)


def test_mit_init():
    path = 'PATH_THAT_DO_NOT_EXIST'
    # Test all combinations of pretrained and init_cfg
    # pretrained=None, init_cfg=None
    model = MixVisionTransformer(pretrained=None, init_cfg=None)
    assert model.init_cfg is None
    model.init_weights()

    # pretrained=None
    # init_cfg loads pretrain from an non-existent file
    model = MixVisionTransformer(
        pretrained=None, init_cfg=dict(type='Pretrained', checkpoint=path))
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained=None
    # init_cfg=123, whose type is unsupported
    model = MixVisionTransformer(pretrained=None, init_cfg=123)
    with pytest.raises(TypeError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg=None
    model = MixVisionTransformer(pretrained=path, init_cfg=None)
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        MixVisionTransformer(
            pretrained=path, init_cfg=dict(type='Pretrained', checkpoint=path))
    with pytest.raises(AssertionError):
        MixVisionTransformer(pretrained=path, init_cfg=123)

    # pretrain=123, whose type is unsupported
    # init_cfg=None
    with pytest.raises(TypeError):
        MixVisionTransformer(pretrained=123, init_cfg=None)

    # pretrain=123, whose type is unsupported
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        MixVisionTransformer(
            pretrained=123, init_cfg=dict(type='Pretrained', checkpoint=path))

    # pretrain=123, whose type is unsupported
    # init_cfg=123, whose type is unsupported
    with pytest.raises(AssertionError):
        MixVisionTransformer(pretrained=123, init_cfg=123)
