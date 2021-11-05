# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.necks import JPU


def test_fastfcn_neck():
    # Test FastFCN Standard Forward
    model = JPU(
        in_channels=(64, 128, 256),
        mid_channels=64,
        start_level=0,
        end_level=-1,
        dilations=(1, 2, 4, 8),
    )
    model.init_weights()
    model.train()
    batch_size = 1
    input = [
        torch.randn(batch_size, 64, 64, 128),
        torch.randn(batch_size, 128, 32, 64),
        torch.randn(batch_size, 256, 16, 32)
    ]
    feat = model(input)

    assert len(feat) == 3
    assert feat[0].shape == torch.Size([batch_size, 64, 64, 128])
    assert feat[1].shape == torch.Size([batch_size, 128, 32, 64])
    assert feat[2].shape == torch.Size([batch_size, 256, 64, 128])

    with pytest.raises(AssertionError):
        # FastFCN input and in_channels constraints.
        JPU(in_channels=(256, 64, 128), start_level=0, end_level=5)

    # Test not default start_level
    model = JPU(in_channels=(64, 128, 256), start_level=1, end_level=-1)
    input = [
        torch.randn(batch_size, 64, 64, 128),
        torch.randn(batch_size, 128, 32, 64),
        torch.randn(batch_size, 256, 16, 32)
    ]
    feat = model(input)
    assert len(feat) == 2
    assert feat[0].shape == torch.Size([batch_size, 128, 32, 64])
    assert feat[1].shape == torch.Size([batch_size, 2048, 32, 64])
