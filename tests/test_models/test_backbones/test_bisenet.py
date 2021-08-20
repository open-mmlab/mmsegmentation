# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import BiSeNetV2


def test_fastscnn_backbone():
    # Test BiSeNetV2 Standard Forward
    model = BiSeNetV2()
    model.init_weights()
    model.train()
    batch_size = 4
    imgs = torch.randn(batch_size, 3, 1024, 2048)
    feat = model(imgs)

    assert len(feat) == 5
    # output for Segment Head
    assert feat[0].shape == torch.Size([batch_size, 128, 128, 256])
    # feat_2
    assert feat[1].shape == torch.Size([batch_size, 16, 256, 512])
    # feat_3
    assert feat[2].shape == torch.Size([batch_size, 32, 128, 256])
    # feat_4
    assert feat[3].shape == torch.Size([batch_size, 64, 64, 128])
    # feat_5_4
    assert feat[4].shape == torch.Size([batch_size, 128, 32, 64])
