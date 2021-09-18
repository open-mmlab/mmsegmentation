# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.necks import JPU


def test_fastfcn_neck():
    # Test FastFCN Standard Forward
    model = JPU()
    model.init_weights()
    model.train()
    batch_size = 1
    input = [
        torch.randn(batch_size, 256, 256, 512),
        torch.randn(batch_size, 512, 128, 256),
        torch.randn(batch_size, 1024, 64, 128),
        torch.randn(batch_size, 2048, 32, 64)
    ]
    feat = model(input)

    assert len(feat) == 4
    assert feat[0].shape == torch.Size([batch_size, 256, 256, 512])
    assert feat[1].shape == torch.Size([batch_size, 512, 128, 256])
    assert feat[2].shape == torch.Size([batch_size, 1024, 64, 128])
    assert feat[3].shape == torch.Size([batch_size, 2048, 128, 256])
