import torch

from mmseg.models import FPN


def test_fpn():
    in_channels = [256, 512, 1024, 2048]
    inputs = [
        torch.randn(1, c, 56 // 2**i, 56 // 2**i)
        for i, c in enumerate(in_channels)
    ]

    fpn = FPN(in_channels, 256, len(in_channels))
    outputs = fpn(inputs)
    assert outputs[0].shape == torch.Size([1, 256, 56, 56])
    assert outputs[1].shape == torch.Size([1, 256, 28, 28])
    assert outputs[2].shape == torch.Size([1, 256, 14, 14])
    assert outputs[3].shape == torch.Size([1, 256, 7, 7])
