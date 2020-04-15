import pytest
import torch
from torch import nn

from mmseg.models.decode_heads import (ASPPHead, FCNHead, GCHead, NLHead,
                                       PSPHead)
from mmseg.models.decode_heads.decode_head import DecodeHead
from mmseg.ops import ConvModule


def _conv_has_norm(module, sync_bn):
    for m in module.modules():
        if isinstance(m, ConvModule):
            if not m.with_norm:
                return False
            if sync_bn:
                if not isinstance(m.bn, nn.SyncBatchNorm):
                    return False
    return True


def test_decode_head():

    with pytest.raises(AssertionError):
        # default input_transform doesn't accept multiple inputs
        DecodeHead([32, 16], 16)

    with pytest.raises(AssertionError):
        # default input_transform doesn't accept multiple inputs
        DecodeHead(32, 16, in_index=[-1, -2])

    with pytest.raises(AssertionError):
        # supported mode is resize_concat only
        DecodeHead(32, 16, input_transform='concat')

    with pytest.raises(AssertionError):
        # in_channels should be list|tuple
        DecodeHead(32, 16, input_transform='resize_concat')

    with pytest.raises(AssertionError):
        # in_index should be list|tuple
        DecodeHead([32], 16, in_index=-1, input_transform='resize_concat')

    with pytest.raises(AssertionError):
        # len(in_index) should equal len(in_channels)
        DecodeHead([32, 16],
                   16,
                   in_index=[-1],
                   input_transform='resize_concat')

    # test default dropout
    head = DecodeHead(32, 16)
    assert hasattr(head, 'dropout') and head.dropout.p == 0.1

    # test set dropout
    head = DecodeHead(32, 16, drop_out_ratio=0.2)
    assert hasattr(head, 'dropout') and head.dropout.p == 0.2

    # test no input_transform
    inputs = [torch.randn(1, 32, 40, 40)]
    head = DecodeHead(32, 16)
    assert head.in_channels == 32
    assert head.input_transform is None
    transformed_inputs = head._transform_inputs(inputs)
    assert transformed_inputs.shape == (1, 32, 40, 40)

    # test input_transform = resize_concat
    inputs = [torch.randn(1, 32, 40, 40), torch.randn(1, 16, 20, 20)]
    head = DecodeHead([32, 16],
                      16,
                      in_index=[0, 1],
                      input_transform='resize_concat')
    assert head.in_channels == 48
    assert head.input_transform == 'resize_concat'
    transformed_inputs = head._transform_inputs(inputs)
    assert transformed_inputs.shape == (1, 48, 40, 40)


def test_fcn_head():

    with pytest.raises(AssertionError):
        # num_convs must be larger than 0
        FCNHead(num_convs=0)

    # test no norm_cfg
    head = FCNHead(in_channels=32, channels=16)
    for m in head.modules():
        if isinstance(m, ConvModule):
            assert not m.with_norm

    # test with norm_cfg
    head = FCNHead(in_channels=32, channels=16, norm_cfg=dict(type='SyncBN'))
    for m in head.modules():
        if isinstance(m, ConvModule):
            assert m.with_norm and isinstance(m.bn, nn.SyncBatchNorm)

    # test concat_input=False
    inputs = [torch.randn(1, 32, 40, 40)]
    head = FCNHead(in_channels=32, channels=16, concat_input=False)
    assert len(head.convs) == 2
    assert not head.concat_input and not hasattr(head, 'conv_cat')
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 40, 40)

    # test concat_input=True
    inputs = [torch.randn(1, 32, 40, 40)]
    head = FCNHead(in_channels=32, channels=16, concat_input=True)
    assert len(head.convs) == 2
    assert head.concat_input
    assert head.conv_cat.in_channels == 48
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 40, 40)

    # test kernel_size=3
    inputs = [torch.randn(1, 32, 40, 40)]
    head = FCNHead(in_channels=32, channels=16)
    for i in range(len(head.convs)):
        assert head.convs[i].kernel_size == (3, 3)
        assert head.convs[i].padding == (1, 1)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 40, 40)

    # test kernel_size=1
    inputs = [torch.randn(1, 32, 40, 40)]
    head = FCNHead(in_channels=32, channels=16, kernel_size=1)
    for i in range(len(head.convs)):
        assert head.convs[i].kernel_size == (1, 1)
        assert head.convs[i].padding == (0, 0)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 40, 40)

    # test num_conv
    inputs = [torch.randn(1, 32, 40, 40)]
    head = FCNHead(in_channels=32, channels=16, num_convs=1)
    assert len(head.convs) == 1
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 40, 40)


def test_psp_head():

    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        PSPHead(in_channels=32, channels=16, pool_scales=1)

    # test no norm_cfg
    head = PSPHead(in_channels=32, channels=16)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = PSPHead(in_channels=32, channels=16, norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 32, 40, 40)]
    head = PSPHead(in_channels=32, channels=16, pool_scales=(1, 2, 3))
    assert head.psp_modules[0][0].output_size == 1
    assert head.psp_modules[1][0].output_size == 2
    assert head.psp_modules[2][0].output_size == 3
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 40, 40)


def test_aspp_head():

    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        ASPPHead(in_channels=32, channels=16, dilations=1)

    # test no norm_cfg
    head = ASPPHead(in_channels=32, channels=16)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = ASPPHead(in_channels=32, channels=16, norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 32, 40, 40)]
    head = ASPPHead(in_channels=32, channels=16, dilations=(1, 12, 24))
    assert head.aspp_modules[0].conv.dilation == (1, 1)
    assert head.aspp_modules[1].conv.dilation == (12, 12)
    assert head.aspp_modules[2].conv.dilation == (24, 24)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 40, 40)


def test_gc_head():
    head = GCHead(in_channels=32, channels=16)
    assert len(head.convs) == 2
    assert hasattr(head, 'gc_block')
    inputs = [torch.randn(1, 32, 40, 40)]
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 40, 40)


def test_nl_head():
    head = NLHead(in_channels=32, channels=16)
    assert len(head.convs) == 2
    assert hasattr(head, 'nl_block')
    inputs = [torch.randn(1, 32, 40, 40)]
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 40, 40)
