from unittest.mock import patch

import pytest
import torch
from mmcv.cnn import ConvModule
from mmcv.utils.parrots_wrapper import SyncBatchNorm

from mmseg.models.decode_heads import (ANNHead, ASPPHead, CCHead, DAHead,
                                       FCNHead, GCHead, NLHead, OCRHead,
                                       PSAHead, PSPHead, UPerHead)
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


def _conv_has_norm(module, sync_bn):
    for m in module.modules():
        if isinstance(m, ConvModule):
            if not m.with_norm:
                return False
            if sync_bn:
                if not isinstance(m.bn, SyncBatchNorm):
                    return False
    return True


def to_cuda(module, data):
    module = module.cuda()
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = data[i].cuda()
    return module, data


@patch.multiple(BaseDecodeHead, __abstractmethods__=set())
def test_decode_head():

    with pytest.raises(AssertionError):
        # default input_transform doesn't accept multiple inputs
        BaseDecodeHead([32, 16], 16)

    with pytest.raises(AssertionError):
        # default input_transform doesn't accept multiple inputs
        BaseDecodeHead(32, 16, in_index=[-1, -2])

    with pytest.raises(AssertionError):
        # supported mode is resize_concat only
        BaseDecodeHead(32, 16, input_transform='concat')

    with pytest.raises(AssertionError):
        # in_channels should be list|tuple
        BaseDecodeHead(32, 16, input_transform='resize_concat')

    with pytest.raises(AssertionError):
        # in_index should be list|tuple
        BaseDecodeHead([32], 16, in_index=-1, input_transform='resize_concat')

    with pytest.raises(AssertionError):
        # len(in_index) should equal len(in_channels)
        BaseDecodeHead([32, 16],
                       16,
                       in_index=[-1],
                       input_transform='resize_concat')

    # test default dropout
    head = BaseDecodeHead(32, 16)
    assert hasattr(head, 'dropout') and head.dropout.p == 0.1

    # test set dropout
    head = BaseDecodeHead(32, 16, drop_out_ratio=0.2)
    assert hasattr(head, 'dropout') and head.dropout.p == 0.2

    # test no input_transform
    inputs = [torch.randn(1, 32, 45, 45)]
    head = BaseDecodeHead(32, 16)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.in_channels == 32
    assert head.input_transform is None
    transformed_inputs = head._transform_inputs(inputs)
    assert transformed_inputs.shape == (1, 32, 45, 45)

    # test input_transform = resize_concat
    inputs = [torch.randn(1, 32, 45, 45), torch.randn(1, 16, 21, 21)]
    head = BaseDecodeHead([32, 16],
                          16,
                          in_index=[0, 1],
                          input_transform='resize_concat')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.in_channels == 48
    assert head.input_transform == 'resize_concat'
    transformed_inputs = head._transform_inputs(inputs)
    assert transformed_inputs.shape == (1, 48, 45, 45)


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
            assert m.with_norm and isinstance(m.bn, SyncBatchNorm)

    # test concat_input=False
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(in_channels=32, channels=16, concat_input=False)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert len(head.convs) == 2
    assert not head.concat_input and not hasattr(head, 'conv_cat')
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test concat_input=True
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(in_channels=32, channels=16, concat_input=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert len(head.convs) == 2
    assert head.concat_input
    assert head.conv_cat.in_channels == 48
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test kernel_size=3
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(in_channels=32, channels=16)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    for i in range(len(head.convs)):
        assert head.convs[i].kernel_size == (3, 3)
        assert head.convs[i].padding == 1
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test kernel_size=1
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(in_channels=32, channels=16, kernel_size=1)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    for i in range(len(head.convs)):
        assert head.convs[i].kernel_size == (1, 1)
        assert head.convs[i].padding == 0
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test num_conv
    inputs = [torch.randn(1, 32, 45, 45)]
    head = FCNHead(in_channels=32, channels=16, num_convs=1)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert len(head.convs) == 1
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


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

    inputs = [torch.randn(1, 32, 45, 45)]
    head = PSPHead(in_channels=32, channels=16, pool_scales=(1, 2, 3))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.psp_modules[0][0].output_size == 1
    assert head.psp_modules[1][0].output_size == 2
    assert head.psp_modules[2][0].output_size == 3
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


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

    inputs = [torch.randn(1, 32, 45, 45)]
    head = ASPPHead(in_channels=32, channels=16, dilations=(1, 12, 24))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.aspp_modules[0].conv.dilation == (1, 1)
    assert head.aspp_modules[1].conv.dilation == (12, 12)
    assert head.aspp_modules[2].conv.dilation == (24, 24)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_psa_head():

    with pytest.raises(AssertionError):
        # psa_type must be in 'bi-direction', 'collect', 'distribute'
        PSAHead(
            in_channels=32, channels=16, mask_size=(39, 39), psa_type='gather')

    # test no norm_cfg
    head = PSAHead(in_channels=32, channels=16, mask_size=(39, 39))
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = PSAHead(
        in_channels=32,
        channels=16,
        mask_size=(39, 39),
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(in_channels=32, channels=16, mask_size=(39, 39))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)


def test_gc_head():
    head = GCHead(in_channels=32, channels=16)
    assert len(head.convs) == 2
    assert hasattr(head, 'gc_block')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_nl_head():
    head = NLHead(in_channels=32, channels=16)
    assert len(head.convs) == 2
    assert hasattr(head, 'nl_block')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_cc_head():
    head = CCHead(in_channels=32, channels=16)
    assert len(head.convs) == 2
    assert hasattr(head, 'cca')
    if torch.cuda.is_available():
        inputs = [torch.randn(1, 32, 45, 45)]
        head, inputs = to_cuda(head, inputs)
        outputs = head(inputs)
        assert outputs.shape == (1, head.num_classes, 45, 45)


def test_uper_head():

    with pytest.raises(AssertionError):
        # fpn_in_channels must be list|tuple
        UPerHead(in_channels=32, channels=16)

    # test no norm_cfg
    head = UPerHead(in_channels=[32, 16], channels=16, in_index=[-2, -1])
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = UPerHead(
        in_channels=[32, 16],
        channels=16,
        norm_cfg=dict(type='SyncBN'),
        in_index=[-2, -1])
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 32, 45, 45), torch.randn(1, 16, 21, 21)]
    head = UPerHead(in_channels=[32, 16], channels=16, in_index=[-2, -1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_ann_head():

    inputs = [torch.randn(1, 16, 45, 45), torch.randn(1, 32, 21, 21)]
    head = ANNHead(
        in_channels=[16, 32],
        channels=16,
        in_index=[-2, -1],
        project_channels=8)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 21, 21)


def test_da_head():

    inputs = [torch.randn(1, 32, 45, 45)]
    head = DAHead(in_channels=32, channels=16, pam_channels=8)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    for output in outputs:
        assert output.shape == (1, head.num_classes, 45, 45)


def test_ocr_head():

    inputs = [torch.randn(1, 32, 45, 45)]
    ocr_head = OCRHead(in_channels=32, channels=16, ocr_channels=8)
    fcn_head = FCNHead(in_channels=32, channels=16)
    if torch.cuda.is_available():
        head, inputs = to_cuda(ocr_head, inputs)
        head, inputs = to_cuda(fcn_head, inputs)
    prev_output = fcn_head(inputs)
    output = ocr_head(inputs, prev_output)
    assert output.shape == (1, ocr_head.num_classes, 45, 45)
