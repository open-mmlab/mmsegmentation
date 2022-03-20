# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models import FPN


def test_fpn():
    in_channels = [64, 128, 256, 512]
    inputs = [
        torch.randn(1, c, 56 // 2**i, 56 // 2**i)
        for i, c in enumerate(in_channels)
    ]

    fpn = FPN(in_channels, 64, len(in_channels))
    outputs = fpn(inputs)
    assert outputs[0].shape == torch.Size([1, 64, 56, 56])
    assert outputs[1].shape == torch.Size([1, 64, 28, 28])
    assert outputs[2].shape == torch.Size([1, 64, 14, 14])
    assert outputs[3].shape == torch.Size([1, 64, 7, 7])

    fpn = FPN(
        in_channels,
        64,
        len(in_channels),
        upsample_cfg=dict(mode='nearest', scale_factor=2.0))
    outputs = fpn(inputs)
    assert outputs[0].shape == torch.Size([1, 64, 56, 56])
    assert outputs[1].shape == torch.Size([1, 64, 28, 28])
    assert outputs[2].shape == torch.Size([1, 64, 14, 14])
    assert outputs[3].shape == torch.Size([1, 64, 7, 7])
