# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import STDCContextPathNet
from mmseg.models.backbones.stdc import (AttentionRefinementModule,
                                         FeatureFusionModule, STDCModule,
                                         STDCNet)


def test_stdc_context_path_net():
    # Test STDCContextPathNet Standard Forward
    model = STDCContextPathNet(
        backbone_cfg=dict(
            type='STDCNet',
            stdc_type='STDCNet1',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            with_final_conv=True),
        last_in_channels=(1024, 512),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4))
    model.init_weights()
    model.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 256, 512)
    feat = model(imgs)

    assert len(feat) == 4
    # output for segment Head
    assert feat[0].shape == torch.Size([batch_size, 256, 32, 64])
    # for auxiliary head 1
    assert feat[1].shape == torch.Size([batch_size, 128, 16, 32])
    # for auxiliary head 2
    assert feat[2].shape == torch.Size([batch_size, 128, 32, 64])
    # for auxiliary head 3
    assert feat[3].shape == torch.Size([batch_size, 256, 32, 64])

    # Test input with rare shape
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 527, 279)
    model = STDCContextPathNet(
        backbone_cfg=dict(
            type='STDCNet',
            stdc_type='STDCNet1',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='add',
            num_convs=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            with_final_conv=False),
        last_in_channels=(1024, 512),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4))
    model.init_weights()
    model.train()
    feat = model(imgs)
    assert len(feat) == 4


def test_stdcnet():
    with pytest.raises(AssertionError):
        # STDC backbone constraints.
        STDCNet(
            stdc_type='STDCNet3',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            with_final_conv=False)

    with pytest.raises(AssertionError):
        # STDC bottleneck type constraints.
        STDCNet(
            stdc_type='STDCNet1',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='dog',
            num_convs=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            with_final_conv=False)

    with pytest.raises(AssertionError):
        # STDC channels length constraints.
        STDCNet(
            stdc_type='STDCNet1',
            in_channels=3,
            channels=(16, 32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            with_final_conv=False)


def test_feature_fusion_module():
    x_ffm = FeatureFusionModule(in_channels=64, out_channels=32)
    assert x_ffm.conv0.in_channels == 64
    assert x_ffm.attention[1].in_channels == 32
    assert x_ffm.attention[2].in_channels == 8
    assert x_ffm.attention[2].out_channels == 32
    x1 = torch.randn(2, 32, 32, 64)
    x2 = torch.randn(2, 32, 32, 64)
    x_out = x_ffm(x1, x2)
    assert x_out.shape == torch.Size([2, 32, 32, 64])


def test_attention_refinement_module():
    x_arm = AttentionRefinementModule(128, 32)
    assert x_arm.conv_layer.in_channels == 128
    assert x_arm.atten_conv_layer[1].conv.out_channels == 32
    x = torch.randn(2, 128, 32, 64)
    x_out = x_arm(x)
    assert x_out.shape == torch.Size([2, 32, 32, 64])


def test_stdc_module():
    x_stdc = STDCModule(in_channels=32, out_channels=32, stride=4)
    assert x_stdc.layers[0].conv.in_channels == 32
    assert x_stdc.layers[3].conv.out_channels == 4
    x = torch.randn(2, 32, 32, 64)
    x_out = x_stdc(x)
    assert x_out.shape == torch.Size([2, 32, 32, 64])
