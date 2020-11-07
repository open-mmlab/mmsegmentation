from unittest.mock import patch

import pytest
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.utils import ConfigDict
from mmcv.utils.parrots_wrapper import SyncBatchNorm

from mmseg.models.decode_heads import (ANNHead, ASPPHead, CCHead, DAHead,
                                       DepthwiseSeparableASPPHead,
                                       DepthwiseSeparableFCNHead, DNLHead,
                                       EMAHead, EncHead, FCNHead, GCHead,
                                       NLHead, OCRHead, PointHead, PSAHead,
                                       PSPHead, UPerHead)
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
        BaseDecodeHead([32, 16], 16, num_classes=19)

    with pytest.raises(AssertionError):
        # default input_transform doesn't accept multiple inputs
        BaseDecodeHead(32, 16, num_classes=19, in_index=[-1, -2])

    with pytest.raises(AssertionError):
        # supported mode is resize_concat only
        BaseDecodeHead(32, 16, num_classes=19, input_transform='concat')

    with pytest.raises(AssertionError):
        # in_channels should be list|tuple
        BaseDecodeHead(32, 16, num_classes=19, input_transform='resize_concat')

    with pytest.raises(AssertionError):
        # in_index should be list|tuple
        BaseDecodeHead([32],
                       16,
                       in_index=-1,
                       num_classes=19,
                       input_transform='resize_concat')

    with pytest.raises(AssertionError):
        # len(in_index) should equal len(in_channels)
        BaseDecodeHead([32, 16],
                       16,
                       num_classes=19,
                       in_index=[-1],
                       input_transform='resize_concat')

    # test default dropout
    head = BaseDecodeHead(32, 16, num_classes=19)
    assert hasattr(head, 'dropout') and head.dropout.p == 0.1

    # test set dropout
    head = BaseDecodeHead(32, 16, num_classes=19, dropout_ratio=0.2)
    assert hasattr(head, 'dropout') and head.dropout.p == 0.2

    # test no input_transform
    inputs = [torch.randn(1, 32, 45, 45)]
    head = BaseDecodeHead(32, 16, num_classes=19)
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
                          num_classes=19,
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


def test_psp_head():

    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        PSPHead(in_channels=32, channels=16, num_classes=19, pool_scales=1)

    # test no norm_cfg
    head = PSPHead(in_channels=32, channels=16, num_classes=19)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = PSPHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 32, 45, 45)]
    head = PSPHead(
        in_channels=32, channels=16, num_classes=19, pool_scales=(1, 2, 3))
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


def test_psa_head():

    with pytest.raises(AssertionError):
        # psa_type must be in 'bi-direction', 'collect', 'distribute'
        PSAHead(
            in_channels=32,
            channels=16,
            num_classes=19,
            mask_size=(39, 39),
            psa_type='gather')

    # test no norm_cfg
    head = PSAHead(
        in_channels=32, channels=16, num_classes=19, mask_size=(39, 39))
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    # test 'bi-direction' psa_type
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32, channels=16, num_classes=19, mask_size=(39, 39))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'bi-direction' psa_type, shrink_factor=1
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        shrink_factor=1)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'bi-direction' psa_type with soft_max
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        psa_softmax=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'collect' psa_type
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        psa_type='collect')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'collect' psa_type, shrink_factor=1
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        shrink_factor=1,
        psa_type='collect')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'collect' psa_type, shrink_factor=1, compact=True
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        psa_type='collect',
        shrink_factor=1,
        compact=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'distribute' psa_type
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        psa_type='distribute')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)


def test_gc_head():
    head = GCHead(in_channels=32, channels=16, num_classes=19)
    assert len(head.convs) == 2
    assert hasattr(head, 'gc_block')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_nl_head():
    head = NLHead(in_channels=32, channels=16, num_classes=19)
    assert len(head.convs) == 2
    assert hasattr(head, 'nl_block')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_cc_head():
    head = CCHead(in_channels=32, channels=16, num_classes=19)
    assert len(head.convs) == 2
    assert hasattr(head, 'cca')
    if not torch.cuda.is_available():
        pytest.skip('CCHead requires CUDA')
    inputs = [torch.randn(1, 32, 45, 45)]
    head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


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


def test_ann_head():

    inputs = [torch.randn(1, 16, 45, 45), torch.randn(1, 32, 21, 21)]
    head = ANNHead(
        in_channels=[16, 32],
        channels=16,
        num_classes=19,
        in_index=[-2, -1],
        project_channels=8)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 21, 21)


def test_da_head():

    inputs = [torch.randn(1, 32, 45, 45)]
    head = DAHead(in_channels=32, channels=16, num_classes=19, pam_channels=8)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, tuple) and len(outputs) == 3
    for output in outputs:
        assert output.shape == (1, head.num_classes, 45, 45)
    test_output = head.forward_test(inputs, None, None)
    assert test_output.shape == (1, head.num_classes, 45, 45)


def test_ocr_head():

    inputs = [torch.randn(1, 32, 45, 45)]
    ocr_head = OCRHead(
        in_channels=32, channels=16, num_classes=19, ocr_channels=8)
    fcn_head = FCNHead(in_channels=32, channels=16, num_classes=19)
    if torch.cuda.is_available():
        head, inputs = to_cuda(ocr_head, inputs)
        head, inputs = to_cuda(fcn_head, inputs)
    prev_output = fcn_head(inputs)
    output = ocr_head(inputs, prev_output)
    assert output.shape == (1, ocr_head.num_classes, 45, 45)


def test_enc_head():
    # with se_loss, w.o. lateral
    inputs = [torch.randn(1, 32, 21, 21)]
    head = EncHead(
        in_channels=[32], channels=16, num_classes=19, in_index=[-1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    assert outputs[0].shape == (1, head.num_classes, 21, 21)
    assert outputs[1].shape == (1, head.num_classes)

    # w.o se_loss, w.o. lateral
    inputs = [torch.randn(1, 32, 21, 21)]
    head = EncHead(
        in_channels=[32],
        channels=16,
        use_se_loss=False,
        num_classes=19,
        in_index=[-1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 21, 21)

    # with se_loss, with lateral
    inputs = [torch.randn(1, 16, 45, 45), torch.randn(1, 32, 21, 21)]
    head = EncHead(
        in_channels=[16, 32],
        channels=16,
        add_lateral=True,
        num_classes=19,
        in_index=[-2, -1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    assert outputs[0].shape == (1, head.num_classes, 21, 21)
    assert outputs[1].shape == (1, head.num_classes)
    test_output = head.forward_test(inputs, None, None)
    assert test_output.shape == (1, head.num_classes, 21, 21)


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


def test_dnl_head():
    # DNL with 'embedded_gaussian' mode
    head = DNLHead(in_channels=32, channels=16, num_classes=19)
    assert len(head.convs) == 2
    assert hasattr(head, 'dnl_block')
    assert head.dnl_block.temperature == 0.05
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # NonLocal2d with 'dot_product' mode
    head = DNLHead(
        in_channels=32, channels=16, num_classes=19, mode='dot_product')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # NonLocal2d with 'gaussian' mode
    head = DNLHead(
        in_channels=32, channels=16, num_classes=19, mode='gaussian')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # NonLocal2d with 'concatenation' mode
    head = DNLHead(
        in_channels=32, channels=16, num_classes=19, mode='concatenation')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_emanet_head():
    head = EMAHead(
        in_channels=32,
        ema_channels=24,
        channels=16,
        num_stages=3,
        num_bases=16,
        num_classes=19)
    for param in head.ema_mid_conv.parameters():
        assert not param.requires_grad
    assert hasattr(head, 'ema_module')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_point_head():

    inputs = [torch.randn(1, 32, 45, 45)]
    point_head = PointHead(
        in_channels=[32], in_index=[0], channels=16, num_classes=19)
    assert len(point_head.fcs) == 3
    fcn_head = FCNHead(in_channels=32, channels=16, num_classes=19)
    if torch.cuda.is_available():
        head, inputs = to_cuda(point_head, inputs)
        head, inputs = to_cuda(fcn_head, inputs)
    prev_output = fcn_head(inputs)
    test_cfg = ConfigDict(
        subdivision_steps=2, subdivision_num_points=8196, scale_factor=2)
    output = point_head.forward_test(inputs, prev_output, None, test_cfg)
    assert output.shape == (1, point_head.num_classes, 180, 180)
