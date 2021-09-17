# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import ICHead
from .utils import _conv_has_norm, to_cuda


def test_ic_head():
    # test no norm_cfg
    head = ICHead(
        in_channels=(64, 256, 256),
        in_index=(0, 1, 2),
        input_transform='multiple_select',
        channels=128,
        dropout_ratio=0,
        norm_cfg=None,
        align_corners=False,
        num_classes=19)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = ICHead(
        in_channels=(64, 256, 256),
        in_index=(0, 1, 2),
        input_transform='multiple_select',
        channels=128,
        dropout_ratio=0,
        norm_cfg=dict(type='SyncBN'),
        align_corners=False,
        num_classes=19)
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [
        torch.randn(1, 64, 128, 256),
        torch.randn(1, 256, 65, 129),
        torch.randn(1, 256, 32, 64)
    ]
    head = ICHead(
        in_channels=(64, 256, 256),
        in_index=(0, 1, 2),
        input_transform='multiple_select',
        channels=128,
        dropout_ratio=0,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        num_classes=19)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs[0].shape == (1, head.num_classes, 1024, 2048)
    assert outputs[1].shape == (1, head.num_classes, 256, 512)
    assert outputs[2].shape == (1, head.num_classes, 128, 256)
    assert outputs[3].shape == (1, head.num_classes, 65, 129)
