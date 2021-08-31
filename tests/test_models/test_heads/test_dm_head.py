# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.decode_heads import DMHead
from .utils import _conv_has_norm, to_cuda


def test_dm_head():

    with pytest.raises(AssertionError):
        # filter_sizes must be list|tuple
        DMHead(in_channels=32, channels=16, num_classes=19, filter_sizes=1)

    # test no norm_cfg
    head = DMHead(in_channels=32, channels=16, num_classes=19)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = DMHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    # fusion=True
    inputs = [torch.randn(1, 32, 45, 45)]
    head = DMHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        filter_sizes=(1, 3, 5),
        fusion=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.fusion is True
    assert head.dcm_modules[0].filter_size == 1
    assert head.dcm_modules[1].filter_size == 3
    assert head.dcm_modules[2].filter_size == 5
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # fusion=False
    inputs = [torch.randn(1, 32, 45, 45)]
    head = DMHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        filter_sizes=(1, 3, 5),
        fusion=False)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.fusion is False
    assert head.dcm_modules[0].filter_size == 1
    assert head.dcm_modules[1].filter_size == 3
    assert head.dcm_modules[2].filter_size == 5
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
