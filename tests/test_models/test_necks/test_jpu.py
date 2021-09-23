# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.necks import JPU


def test_fastfcn_neck():
    # Test FastFCN Standard Forward
    model = JPU()
    model.init_weights()
    model.train()
    batch_size = 1
    input = [
        torch.randn(batch_size, 256, 128, 256),
        torch.randn(batch_size, 512, 64, 128),
        torch.randn(batch_size, 1024, 32, 64),
        torch.randn(batch_size, 2048, 16, 32)
    ]
    feat = model(input)

    assert len(feat) == 4
    assert feat[0].shape == torch.Size([batch_size, 256, 128, 256])
    assert feat[1].shape == torch.Size([batch_size, 512, 64, 128])
    assert feat[2].shape == torch.Size([batch_size, 1024, 32, 64])
    assert feat[3].shape == torch.Size([batch_size, 2048, 64, 128])

    with pytest.raises(AssertionError):
        # FastFCN input and in_channels constraints.
        JPU(in_channels=(128, 256, 512, 1024), start_level=1, end_level=5)
