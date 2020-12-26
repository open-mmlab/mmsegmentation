import mmcv
import pytest
import torch

from mmseg.models.utils.se_layer import SELayer


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
