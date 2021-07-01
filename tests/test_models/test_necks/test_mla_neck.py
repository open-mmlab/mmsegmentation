import torch

from mmseg.models import MLANeck


def test_mla():
    in_channels = [1024, 1024, 1024, 1024]
    mla = MLANeck(in_channels, 256)

    inputs = [torch.randn(1, c, 24, 24) for i, c in enumerate(in_channels)]
    outputs = mla(inputs)
    assert outputs[0].shape == torch.Size([1, 256, 24, 24])
    assert outputs[1].shape == torch.Size([1, 256, 24, 24])
    assert outputs[2].shape == torch.Size([1, 256, 24, 24])
    assert outputs[3].shape == torch.Size([1, 256, 24, 24])
