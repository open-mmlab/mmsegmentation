import torch

from mmseg.models.backbones.swin import SwinTransformer


def test_swin_transformer():
    """Test Swin Transformer backbone."""
    # Test abnormal inference size
    temp = torch.randn((1, 3, 511, 511))
    model = SwinTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 96, 128, 128)
    assert outs[1].shape == (1, 192, 64, 64)
    assert outs[2].shape == (1, 384, 32, 32)
    assert outs[3].shape == (1, 768, 16, 16)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 112, 137))
    model = SwinTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 96, 28, 35)
    assert outs[1].shape == (1, 192, 14, 18)
    assert outs[2].shape == (1, 384, 7, 9)
    assert outs[3].shape == (1, 768, 4, 5)

    model = SwinTransformer(frozen_stages=4)
    model.train()
    for p in model.parameters():
        assert not p.requires_grad
