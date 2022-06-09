# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.necks import ICNeck
from mmseg.models.necks.ic_neck import CascadeFeatureFusion
from ..test_heads.utils import _conv_has_norm, to_cuda


def test_ic_neck():
    # test with norm_cfg
    neck = ICNeck(
        in_channels=(4, 16, 16),
        out_channels=8,
        norm_cfg=dict(type='SyncBN'),
        align_corners=False)
    assert _conv_has_norm(neck, sync_bn=True)

    inputs = [
        torch.randn(1, 4, 32, 64),
        torch.randn(1, 16, 16, 32),
        torch.randn(1, 16, 8, 16)
    ]
    neck = ICNeck(
        in_channels=(4, 16, 16),
        out_channels=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False)
    if torch.cuda.is_available():
        neck, inputs = to_cuda(neck, inputs)

    outputs = neck(inputs)
    assert outputs[0].shape == (1, 4, 16, 32)
    assert outputs[1].shape == (1, 4, 32, 64)
    assert outputs[1].shape == (1, 4, 32, 64)


def test_ic_neck_cascade_feature_fusion():
    cff = CascadeFeatureFusion(64, 64, 32)
    assert cff.conv_low.in_channels == 64
    assert cff.conv_low.out_channels == 32
    assert cff.conv_high.in_channels == 64
    assert cff.conv_high.out_channels == 32


def test_ic_neck_input_channels():
    with pytest.raises(AssertionError):
        # ICNet Neck input channel constraints.
        ICNeck(
            in_channels=(16, 64, 64, 64),
            out_channels=32,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False)
