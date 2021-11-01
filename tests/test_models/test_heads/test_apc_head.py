# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.decode_heads import APCHead
from .utils import _conv_has_norm, to_cuda


def test_apc_head():

    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        APCHead(in_channels=8, channels=2, num_classes=19, pool_scales=1)

    # test no norm_cfg
    head = APCHead(in_channels=8, channels=2, num_classes=19)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = APCHead(
        in_channels=8,
        channels=2,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    # fusion=True
    inputs = [torch.randn(1, 8, 45, 45)]
    head = APCHead(
        in_channels=8,
        channels=2,
        num_classes=19,
        pool_scales=(1, 2, 3),
        fusion=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.fusion is True
    assert head.acm_modules[0].pool_scale == 1
    assert head.acm_modules[1].pool_scale == 2
    assert head.acm_modules[2].pool_scale == 3
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # fusion=False
    inputs = [torch.randn(1, 8, 45, 45)]
    head = APCHead(
        in_channels=8,
        channels=2,
        num_classes=19,
        pool_scales=(1, 2, 3),
        fusion=False)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.fusion is False
    assert head.acm_modules[0].pool_scale == 1
    assert head.acm_modules[1].pool_scale == 2
    assert head.acm_modules[2].pool_scale == 3
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
