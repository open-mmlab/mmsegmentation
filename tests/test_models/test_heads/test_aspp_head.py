# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.decode_heads import ASPPHead, DepthwiseSeparableASPPHead
from .utils import _conv_has_norm, to_cuda


def test_aspp_head():

    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        ASPPHead(in_channels=32, channels=16, num_classes=19, dilations=1)

    # test no norm_cfg
    head = ASPPHead(in_channels=32, channels=16, num_classes=19)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = ASPPHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 32, 45, 45)]
    head = ASPPHead(
        in_channels=32, channels=16, num_classes=19, dilations=(1, 12, 24))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.aspp_modules[0].conv.dilation == (1, 1)
    assert head.aspp_modules[1].conv.dilation == (12, 12)
    assert head.aspp_modules[2].conv.dilation == (24, 24)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_dw_aspp_head():

    # test w.o. c1
    inputs = [torch.randn(1, 32, 45, 45)]
    head = DepthwiseSeparableASPPHead(
        c1_in_channels=0,
        c1_channels=0,
        in_channels=32,
        channels=16,
        num_classes=19,
        dilations=(1, 12, 24))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.c1_bottleneck is None
    assert head.aspp_modules[0].conv.dilation == (1, 1)
    assert head.aspp_modules[1].depthwise_conv.dilation == (12, 12)
    assert head.aspp_modules[2].depthwise_conv.dilation == (24, 24)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test with c1
    inputs = [torch.randn(1, 8, 45, 45), torch.randn(1, 32, 21, 21)]
    head = DepthwiseSeparableASPPHead(
        c1_in_channels=8,
        c1_channels=4,
        in_channels=32,
        channels=16,
        num_classes=19,
        dilations=(1, 12, 24))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.c1_bottleneck.in_channels == 8
    assert head.c1_bottleneck.out_channels == 4
    assert head.aspp_modules[0].conv.dilation == (1, 1)
    assert head.aspp_modules[1].depthwise_conv.dilation == (12, 12)
    assert head.aspp_modules[2].depthwise_conv.dilation == (24, 24)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
