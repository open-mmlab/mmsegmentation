# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import FastSCNN


def test_fastscnn_backbone():
    with pytest.raises(AssertionError):
        # Fast-SCNN channel constraints.
        FastSCNN(
            3, (32, 48),
            64, (64, 96, 128), (2, 2, 1),
            global_out_channels=127,
            higher_in_channels=64,
            lower_in_channels=128)

    # Test FastSCNN Standard Forward
    model = FastSCNN()
    model.init_weights()
    model.train()
    batch_size = 4
    imgs = torch.randn(batch_size, 3, 512, 1024)
    feat = model(imgs)

    assert len(feat) == 3
    # higher-res
    assert feat[0].shape == torch.Size([batch_size, 64, 64, 128])
    # lower-res
    assert feat[1].shape == torch.Size([batch_size, 128, 16, 32])
    # FFM output
    assert feat[2].shape == torch.Size([batch_size, 128, 64, 128])
