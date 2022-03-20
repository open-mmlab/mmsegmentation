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
    model = FastSCNN(
        in_channels=3,
        downsample_dw_channels=(4, 6),
        global_in_channels=8,
        global_block_channels=(8, 12, 16),
        global_block_strides=(2, 2, 1),
        global_out_channels=16,
        higher_in_channels=8,
        lower_in_channels=16,
        fusion_out_channels=16,
    )
    model.init_weights()
    model.train()
    batch_size = 4
    imgs = torch.randn(batch_size, 3, 64, 128)
    feat = model(imgs)

    assert len(feat) == 3
    # higher-res
    assert feat[0].shape == torch.Size([batch_size, 8, 8, 16])
    # lower-res
    assert feat[1].shape == torch.Size([batch_size, 16, 2, 4])
    # FFM output
    assert feat[2].shape == torch.Size([batch_size, 16, 8, 16])
