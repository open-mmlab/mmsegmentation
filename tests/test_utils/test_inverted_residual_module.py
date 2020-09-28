import pytest
import torch

from mmseg.models.utils import InvertedResidual


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
