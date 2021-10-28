# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.decode_heads import PSAHead
from .utils import _conv_has_norm, to_cuda


def test_psa_head():

    with pytest.raises(AssertionError):
        # psa_type must be in 'bi-direction', 'collect', 'distribute'
        PSAHead(
            in_channels=4,
            channels=2,
            num_classes=19,
            mask_size=(13, 13),
            psa_type='gather')

    # test no norm_cfg
    head = PSAHead(
        in_channels=4, channels=2, num_classes=19, mask_size=(13, 13))
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    # test 'bi-direction' psa_type
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4, channels=2, num_classes=19, mask_size=(13, 13))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'bi-direction' psa_type, shrink_factor=1
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        shrink_factor=1)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'bi-direction' psa_type with soft_max
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        psa_softmax=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'collect' psa_type
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        psa_type='collect')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'collect' psa_type, shrink_factor=1
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        shrink_factor=1,
        psa_type='collect')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'collect' psa_type, shrink_factor=1, compact=True
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        psa_type='collect',
        shrink_factor=1,
        compact=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'distribute' psa_type
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        psa_type='distribute')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)
