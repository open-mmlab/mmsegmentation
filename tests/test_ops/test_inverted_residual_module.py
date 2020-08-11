import pytest
import torch
import torch.nn as nn

from mmseg.ops import InvertedResidual


def test_inv_residual():
    with pytest.raises(AssertionError):
        # test stride assertion.
        InvertedResidual(32, 32, 3, 4)

    # test default config with res connection.
    # set expand_ratio = 4, stride = 1 and inp=oup.
    inv_module = InvertedResidual(32, 32, 1, 4)
    assert inv_module.use_res_connect
    assert inv_module.conv[0].kernel_size == 3
    assert inv_module.conv[0].padding == 1
    x = torch.rand(1, 32, 64, 64)
    output = inv_module(x)
    assert output.shape == (1, 32, 64, 64)

    # test inv_residual module without res connection.
    # set expand_ratio = 4, stride = 2.
    inv_module = InvertedResidual(32, 32, 2, 4)
    assert not inv_module.use_res_connect
    assert inv_module.conv[0].kernel_size == 1
    x = torch.rand(1, 32, 64, 64)
    output = inv_module(x)
    assert output.shape == (1, 32, 32, 32)





