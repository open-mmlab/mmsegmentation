# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.necks import ICNeck
from ..test_heads.utils import _conv_has_norm, to_cuda


def test_ic_neck():
    # test with norm_cfg
    neck = ICNeck(
        in_channels=(64, 256, 256),
        in_index=(0, 1, 2),
        out_channels=128,
        norm_cfg=dict(type='SyncBN'),
        align_corners=False)
    assert _conv_has_norm(neck, sync_bn=True)

    inputs = [
        torch.randn(1, 64, 128, 256),
        torch.randn(1, 256, 65, 129),
        torch.randn(1, 256, 32, 64)
    ]
    neck = ICNeck(
        in_channels=(64, 256, 256),
        in_index=(0, 1, 2),
        out_channels=128,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False)
    if torch.cuda.is_available():
        neck, inputs = to_cuda(neck, inputs)
    outputs = neck(inputs)
    assert outputs[0].shape == (1, 128, 65, 129)
    assert outputs[1].shape == (1, 128, 128, 256)
    assert outputs[1].shape == (1, 128, 128, 256)
