# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.backbones import BiSeNetV2


def test_bisenetv2_backbone():
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
