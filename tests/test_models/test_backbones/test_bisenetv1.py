# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import BiSeNetV1


def test_bisenetv1_backbone():
    # Test BiSeNetV1 Standard Forward
    backbone_cfg = dict(
        type='ResNet',
        in_channels=3,
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True)
    model = BiSeNetV1(in_channels=3, backbone_cfg=backbone_cfg)
    model.init_weights()
    model.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 512, 1024)
    feat = model(imgs)

    assert len(feat) == 3
    # output for segment Head
    assert feat[0].shape == torch.Size([batch_size, 256, 64, 128])
    # for auxiliary head 1
    assert feat[1].shape == torch.Size([batch_size, 128, 64, 128])
    # for auxiliary head 2
    assert feat[2].shape == torch.Size([batch_size, 128, 32, 64])

    # Test input with rare shape
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 952, 527)
    feat = model(imgs)
    assert len(feat) == 3

    with pytest.raises(AssertionError):
        # BiSeNetV1 spatial path channel constraints.
        BiSeNetV1(
            backbone_cfg=backbone_cfg,
            in_channels=3,
            spatial_channels=(64, 64, 64))

    with pytest.raises(AssertionError):
        # BiSeNetV1 context path constraints.
        BiSeNetV1(
            backbone_cfg=backbone_cfg,
            in_channels=3,
            context_channels=(128, 256, 512, 1024))
