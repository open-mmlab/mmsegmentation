# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.utils.parrots_wrapper import SyncBatchNorm

from mmseg.models.decode_heads import DepthwiseSeparableFCNHead, FCNHead
from .utils import to_cuda


def test_fcn_head():

    with pytest.raises(AssertionError):
        # num_convs must be not less than 0
        FCNHead(num_classes=19, num_convs=-1)

    # test no norm_cfg
    head = FCNHead(in_channels=32, channels=16, num_classes=19)
    for m in head.modules():
        if isinstance(m, ConvModule):
            assert not m.with_norm

    # test with norm_cfg
    head = FCNHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'))
    for m in head.modules():
        if isinstance(m, ConvModule):
            assert m.with_norm and isinstance(m.bn, SyncBatchNorm)

    # test concat_input=False
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(
        in_channels=32, channels=16, num_classes=19, concat_input=False)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert len(head.convs) == 2
    assert not head.concat_input and not hasattr(head, 'conv_cat')
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test concat_input=True
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(
        in_channels=32, channels=16, num_classes=19, concat_input=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert len(head.convs) == 2
    assert head.concat_input
    assert head.conv_cat.in_channels == 48
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test kernel_size=3
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(in_channels=32, channels=16, num_classes=19)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    for i in range(len(head.convs)):
        assert head.convs[i].kernel_size == (3, 3)
        assert head.convs[i].padding == 1
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test kernel_size=1
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(in_channels=32, channels=16, num_classes=19, kernel_size=1)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    for i in range(len(head.convs)):
        assert head.convs[i].kernel_size == (1, 1)
        assert head.convs[i].padding == 0
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test num_conv
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(in_channels=32, channels=16, num_classes=19, num_convs=1)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert len(head.convs) == 1
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test num_conv = 0
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(
        in_channels=32,
        channels=32,
        num_classes=19,
        num_convs=0,
        concat_input=False)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert isinstance(head.convs, torch.nn.Identity)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_sep_fcn_head():
    # test sep_fcn_head with concat_input=False
    head = DepthwiseSeparableFCNHead(
        in_channels=128,
        channels=128,
        concat_input=False,
        num_classes=19,
        in_index=-1,
        norm_cfg=dict(type='BN', requires_grad=True, momentum=0.01))
    x = [torch.rand(2, 128, 32, 32)]
    output = head(x)
    assert output.shape == (2, head.num_classes, 32, 32)
    assert not head.concat_input
    assert isinstance(head.convs[0], DepthwiseSeparableConvModule)
    assert isinstance(head.convs[1], DepthwiseSeparableConvModule)
    assert head.conv_seg.kernel_size == (1, 1)

    head = DepthwiseSeparableFCNHead(
        in_channels=64,
        channels=64,
        concat_input=True,
        num_classes=19,
        in_index=-1,
        norm_cfg=dict(type='BN', requires_grad=True, momentum=0.01))
    x = [torch.rand(3, 64, 32, 32)]
    output = head(x)
    assert output.shape == (3, head.num_classes, 32, 32)
    assert head.concat_input
    assert isinstance(head.convs[0], DepthwiseSeparableConvModule)
    assert isinstance(head.convs[1], DepthwiseSeparableConvModule)
