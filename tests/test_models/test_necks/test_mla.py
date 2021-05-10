import torch

from mmseg.models import MLA


def test_mla():
    in_channels = [1024, 1024, 1024, 1024]
    mla = MLA(in_channels, 256)

    inputs_4d = [torch.randn(1, c, 24, 24) for i, c in enumerate(in_channels)]
    outputs = mla(inputs_4d)
    assert outputs[0].shape == torch.Size([1, 256, 24, 24])
    assert outputs[1].shape == torch.Size([1, 256, 24, 24])
    assert outputs[2].shape == torch.Size([1, 256, 24, 24])
    assert outputs[3].shape == torch.Size([1, 256, 24, 24])

    inputs_3d = [torch.randn(1, 24 * 24, c) for i, c in enumerate(in_channels)]
    outputs = mla(inputs_3d)
    assert outputs[0].shape == torch.Size([1, 256, 24, 24])
    assert outputs[1].shape == torch.Size([1, 256, 24, 24])
    assert outputs[2].shape == torch.Size([1, 256, 24, 24])
    assert outputs[3].shape == torch.Size([1, 256, 24, 24])
