# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models import MLANeck


def test_mla():
    in_channels = [4, 4, 4, 4]
    mla = MLANeck(in_channels, 32)

    inputs = [torch.randn(1, c, 12, 12) for i, c in enumerate(in_channels)]
    outputs = mla(inputs)
    assert outputs[0].shape == torch.Size([1, 32, 12, 12])
    assert outputs[1].shape == torch.Size([1, 32, 12, 12])
    assert outputs[2].shape == torch.Size([1, 32, 12, 12])
    assert outputs[3].shape == torch.Size([1, 32, 12, 12])
