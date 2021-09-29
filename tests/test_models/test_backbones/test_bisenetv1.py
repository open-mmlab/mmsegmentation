# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import BiSeNetV1
from mmseg.models.backbones.bisenetv1 import (AttentionRefinementModule,
                                              ContextPath, FeatureFusionModule,
                                              SpatialPath)


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
    imgs = torch.randn(batch_size, 3, 256, 512)
    feat = model(imgs)

    assert len(feat) == 3
    # output for segment Head
    assert feat[0].shape == torch.Size([batch_size, 256, 32, 64])
    # for auxiliary head 1
    assert feat[1].shape == torch.Size([batch_size, 128, 32, 64])
    # for auxiliary head 2
    assert feat[2].shape == torch.Size([batch_size, 128, 16, 32])

    # Test input with rare shape
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 527, 279)
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


def test_bisenetv1_spatial_path():
    with pytest.raises(AssertionError):
        # BiSeNetV1 spatial path channel constraints.
        SpatialPath(num_channels=(64, 64, 64), in_channels=3)


def test_bisenetv1_context_path():
    backbone_cfg = dict(
        type='ResNet',
        in_channels=3,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True)

    with pytest.raises(AssertionError):
        # BiSeNetV1 context path constraints.
        ContextPath(
            backbone_cfg=backbone_cfg, context_channels=(128, 256, 512, 1024))


def test_bisenetv1_attention_refinement_module():
    x_arm = AttentionRefinementModule(256, 64)
    assert x_arm.conv_layer.in_channels == 256
    assert x_arm.conv_layer.out_channels == 64
    assert x_arm.conv_layer.kernel_size == (3, 3)
    x = torch.randn(2, 256, 32, 64)
    x_out = x_arm(x)
    assert x_out.shape == torch.Size([2, 64, 32, 64])


def test_bisenetv1_feature_fusion_module():
    ffm = FeatureFusionModule(128, 256)
    assert ffm.conv1.in_channels == 128
    assert ffm.conv1.out_channels == 256
    assert ffm.conv1.kernel_size == (1, 1)
    assert ffm.gap.output_size == (1, 1)
    assert ffm.conv_atten[0].in_channels == 256
    assert ffm.conv_atten[0].out_channels == 256
    assert ffm.conv_atten[0].kernel_size == (1, 1)

    ffm = FeatureFusionModule(128, 128)
    x1 = torch.randn(2, 64, 64, 128)
    x2 = torch.randn(2, 64, 64, 128)
    x_out = ffm(x1, x2)
    assert x_out.shape == torch.Size([2, 128, 64, 128])
