# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.decode_heads import UPerHead
from .utils import _conv_has_norm, to_cuda


def test_uper_head():

    with pytest.raises(AssertionError):
        # fpn_in_channels must be list|tuple
        UPerHead(in_channels=32, channels=16, num_classes=19)

    # test no norm_cfg
    head = UPerHead(
        in_channels=[32, 16], channels=16, num_classes=19, in_index=[-2, -1])
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = UPerHead(
        in_channels=[32, 16],
        channels=16,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'),
        in_index=[-2, -1])
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 32, 45, 45), torch.randn(1, 16, 21, 21)]
    head = UPerHead(
        in_channels=[32, 16], channels=16, num_classes=19, in_index=[-2, -1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
