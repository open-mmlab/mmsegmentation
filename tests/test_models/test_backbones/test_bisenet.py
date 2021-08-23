# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import BiSeNetV2


def test_fastscnn_backbone():
    # BiSeNetV2 wrong pretrained path
    with pytest.raises(TypeError):
        BiSeNetV2(
            pretrained=0,
            out_indices=(0, 1, 2, 3, 4),
            detail_branch_channels=(64, 64, 128),
            channel_ratio=0.25,
            expansion_ratio=6,
            align_corners=False,
            middle_channels=128)

    # Test BiSeNetV2 Standard Forward
    model = BiSeNetV2()
    model.init_weights()
    model.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 512, 1024)
    feat = model(imgs)

    assert len(feat) == 5
    # output for Segment Head
    assert feat[0].shape == torch.Size([batch_size, 128, 64, 128])
    # feat_2
    assert feat[1].shape == torch.Size([batch_size, 16, 128, 256])
    # feat_3
    assert feat[2].shape == torch.Size([batch_size, 32, 64, 128])
    # feat_4
    assert feat[3].shape == torch.Size([batch_size, 64, 32, 64])
    # feat_5_4
    assert feat[4].shape == torch.Size([batch_size, 128, 16, 32])
