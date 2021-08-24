# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import pytest
import torch

from mmseg.models.utils import (InvertedResidual, InvertedResidualV3, SELayer,
                                make_divisible)


def test_make_divisible():
    # test with min_value = None
    assert make_divisible(10, 4) == 12
    assert make_divisible(9, 4) == 12
    assert make_divisible(1, 4) == 4

    # test with min_value = 8
    assert make_divisible(10, 4, 8) == 12
    assert make_divisible(9, 4, 8) == 12
    assert make_divisible(1, 4, 8) == 8


def test_inv_residual():
    with pytest.raises(AssertionError):
        # test stride assertion.
        InvertedResidual(32, 32, 3, 4)

    # test default config with res connection.
    # set expand_ratio = 4, stride = 1 and inp=oup.
    inv_module = InvertedResidual(32, 32, 1, 4)
    assert inv_module.use_res_connect
    assert inv_module.conv[0].kernel_size == (1, 1)
    assert inv_module.conv[0].padding == 0
    assert inv_module.conv[1].kernel_size == (3, 3)
    assert inv_module.conv[1].padding == 1
    assert inv_module.conv[0].with_norm
    assert inv_module.conv[1].with_norm
    x = torch.rand(1, 32, 64, 64)
    output = inv_module(x)
    assert output.shape == (1, 32, 64, 64)

    # test inv_residual module without res connection.
    # set expand_ratio = 4, stride = 2.
    inv_module = InvertedResidual(32, 32, 2, 4)
    assert not inv_module.use_res_connect
    assert inv_module.conv[0].kernel_size == (1, 1)
    x = torch.rand(1, 32, 64, 64)
    output = inv_module(x)
    assert output.shape == (1, 32, 32, 32)

    # test expand_ratio == 1
    inv_module = InvertedResidual(32, 32, 1, 1)
    assert inv_module.conv[0].kernel_size == (3, 3)
    x = torch.rand(1, 32, 64, 64)
    output = inv_module(x)
    assert output.shape == (1, 32, 64, 64)

    # test with checkpoint forward
    inv_module = InvertedResidual(32, 32, 1, 1, with_cp=True)
    assert inv_module.with_cp
    x = torch.rand(1, 32, 64, 64, requires_grad=True)
    output = inv_module(x)
    assert output.shape == (1, 32, 64, 64)


def test_inv_residualv3():
    with pytest.raises(AssertionError):
        # test stride assertion.
        InvertedResidualV3(32, 32, 16, stride=3)

    with pytest.raises(AssertionError):
        # test assertion.
        InvertedResidualV3(32, 32, 16, with_expand_conv=False)

    # test with se_cfg=None, with_expand_conv=False
    inv_module = InvertedResidualV3(32, 32, 32, with_expand_conv=False)

    assert inv_module.with_res_shortcut is True
    assert inv_module.with_se is False
    assert inv_module.with_expand_conv is False
    assert not hasattr(inv_module, 'expand_conv')
    assert isinstance(inv_module.depthwise_conv.conv, torch.nn.Conv2d)
    assert inv_module.depthwise_conv.conv.kernel_size == (3, 3)
    assert inv_module.depthwise_conv.conv.stride == (1, 1)
    assert inv_module.depthwise_conv.conv.padding == (1, 1)
    assert isinstance(inv_module.depthwise_conv.bn, torch.nn.BatchNorm2d)
    assert isinstance(inv_module.depthwise_conv.activate, torch.nn.ReLU)
    assert inv_module.linear_conv.conv.kernel_size == (1, 1)
    assert inv_module.linear_conv.conv.stride == (1, 1)
    assert inv_module.linear_conv.conv.padding == (0, 0)
    assert isinstance(inv_module.linear_conv.bn, torch.nn.BatchNorm2d)

    x = torch.rand(1, 32, 64, 64)
    output = inv_module(x)
    assert output.shape == (1, 32, 64, 64)

    # test with se_cfg and with_expand_conv
    se_cfg = dict(
        channels=16,
        ratio=4,
        act_cfg=(dict(type='ReLU'),
                 dict(type='HSigmoid', bias=3.0, divisor=6.0)))
    act_cfg = dict(type='HSwish')
    inv_module = InvertedResidualV3(
        32, 40, 16, 3, 2, se_cfg=se_cfg, act_cfg=act_cfg)
    assert inv_module.with_res_shortcut is False
    assert inv_module.with_se is True
    assert inv_module.with_expand_conv is True
    assert inv_module.expand_conv.conv.kernel_size == (1, 1)
    assert inv_module.expand_conv.conv.stride == (1, 1)
    assert inv_module.expand_conv.conv.padding == (0, 0)
    assert isinstance(inv_module.expand_conv.activate, mmcv.cnn.HSwish)

    assert isinstance(inv_module.depthwise_conv.conv,
                      mmcv.cnn.bricks.Conv2dAdaptivePadding)
    assert inv_module.depthwise_conv.conv.kernel_size == (3, 3)
    assert inv_module.depthwise_conv.conv.stride == (2, 2)
    assert inv_module.depthwise_conv.conv.padding == (0, 0)
    assert isinstance(inv_module.depthwise_conv.bn, torch.nn.BatchNorm2d)
    assert isinstance(inv_module.depthwise_conv.activate, mmcv.cnn.HSwish)
    assert inv_module.linear_conv.conv.kernel_size == (1, 1)
    assert inv_module.linear_conv.conv.stride == (1, 1)
    assert inv_module.linear_conv.conv.padding == (0, 0)
    assert isinstance(inv_module.linear_conv.bn, torch.nn.BatchNorm2d)
    x = torch.rand(1, 32, 64, 64)
    output = inv_module(x)
    assert output.shape == (1, 40, 32, 32)

    # test with checkpoint forward
    inv_module = InvertedResidualV3(
        32, 40, 16, 3, 2, se_cfg=se_cfg, act_cfg=act_cfg, with_cp=True)
    assert inv_module.with_cp
    x = torch.randn(2, 32, 64, 64, requires_grad=True)
    output = inv_module(x)
    assert output.shape == (2, 40, 32, 32)


def test_se_layer():
    with pytest.raises(AssertionError):
        # test act_cfg assertion.
        SELayer(32, act_cfg=(dict(type='ReLU'), ))

    # test config with channels = 16.
    se_layer = SELayer(16)
    assert se_layer.conv1.conv.kernel_size == (1, 1)
    assert se_layer.conv1.conv.stride == (1, 1)
    assert se_layer.conv1.conv.padding == (0, 0)
    assert isinstance(se_layer.conv1.activate, torch.nn.ReLU)
    assert se_layer.conv2.conv.kernel_size == (1, 1)
    assert se_layer.conv2.conv.stride == (1, 1)
    assert se_layer.conv2.conv.padding == (0, 0)
    assert isinstance(se_layer.conv2.activate, mmcv.cnn.HSigmoid)

    x = torch.rand(1, 16, 64, 64)
    output = se_layer(x)
    assert output.shape == (1, 16, 64, 64)

    # test config with channels = 16, act_cfg = dict(type='ReLU').
    se_layer = SELayer(16, act_cfg=dict(type='ReLU'))
    assert se_layer.conv1.conv.kernel_size == (1, 1)
    assert se_layer.conv1.conv.stride == (1, 1)
    assert se_layer.conv1.conv.padding == (0, 0)
    assert isinstance(se_layer.conv1.activate, torch.nn.ReLU)
    assert se_layer.conv2.conv.kernel_size == (1, 1)
    assert se_layer.conv2.conv.stride == (1, 1)
    assert se_layer.conv2.conv.padding == (0, 0)
    assert isinstance(se_layer.conv2.activate, torch.nn.ReLU)

    x = torch.rand(1, 16, 64, 64)
    output = se_layer(x)
    assert output.shape == (1, 16, 64, 64)
