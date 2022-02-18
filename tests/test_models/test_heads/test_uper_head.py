# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.decode_heads import UPerHead
from .utils import _conv_has_norm, to_cuda


def test_uper_head():

    with pytest.raises(AssertionError):
        # fpn_in_channels must be list|tuple
        UPerHead(in_channels=4, channels=2, num_classes=19)

    # test no norm_cfg
    head = UPerHead(
        in_channels=[4, 2], channels=2, num_classes=19, in_index=[-2, -1])
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = UPerHead(
        in_channels=[4, 2],
        channels=2,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'),
        in_index=[-2, -1])
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 4, 45, 45), torch.randn(1, 2, 21, 21)]
    head = UPerHead(
        in_channels=[4, 2], channels=2, num_classes=19, in_index=[-2, -1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_kernel_update_forward():
    # test kernel updation
    inputs = [torch.randn(1, 4, 45, 45), torch.randn(1, 2, 21, 21)]
    out_channels = 2
    head = UPerHead(
        in_channels=[4, 2],
        channels=out_channels,
        num_classes=19,
        in_index=[-2, -1],
        kernel_update=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    output, feats, seg_kernels = head(inputs)
    assert output.shape == (1, head.num_classes, 45, 45)
    assert feats.shape == (1, out_channels, 45, 45)
    assert seg_kernels.shape == (1, head.num_classes, out_channels, 1, 1)
