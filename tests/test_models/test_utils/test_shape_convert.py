# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.utils import (nchw2nlc2nchw, nchw_to_nlc, nlc2nchw2nlc,
                                nlc_to_nchw)


def test_nchw2nlc2nchw():
    # Test nchw2nlc2nchw function
    shape_nchw = (4, 2, 5, 5)
    shape_nlc = (4, 25, 2)

    def test_func(x):
        assert x.shape == torch.Size(shape_nlc)
        return x

    x = torch.rand(*shape_nchw)
    output = nchw2nlc2nchw(test_func, x)
    assert output.shape == torch.Size(shape_nchw)

    def test_func2(x, arg):
        assert x.shape == torch.Size(shape_nlc)
        assert arg == 100
        return x

    x = torch.rand(*shape_nchw)
    output = nchw2nlc2nchw(test_func2, x, arg=100)
    assert output.shape == torch.Size(shape_nchw)

    def test_func3(x):
        assert x.is_contiguous()
        assert x.shape == torch.Size(shape_nlc)
        return x

    x = torch.rand(*shape_nchw)
    output = nchw2nlc2nchw(test_func3, x, contiguous=True)
    assert output.shape == torch.Size(shape_nchw)
    assert output.is_contiguous()


def test_nlc2nchw2nlc():
    # Test nlc2nchw2nlc function
    shape_nchw = (4, 2, 5, 5)
    shape_nlc = (4, 25, 2)

    def test_func(x):
        assert x.shape == torch.Size(shape_nchw)
        return x

    x = torch.rand(*shape_nlc)
    output = nlc2nchw2nlc(test_func, x, shape_nchw[2:])
    assert output.shape == torch.Size(shape_nlc)

    def test_func2(x, arg):
        assert x.shape == torch.Size(shape_nchw)
        assert arg == 100
        return x

    x = torch.rand(*shape_nlc)
    output = nlc2nchw2nlc(test_func2, x, shape_nchw[2:], arg=100)
    assert output.shape == torch.Size(shape_nlc)

    def test_func3(x):
        assert x.is_contiguous()
        assert x.shape == torch.Size(shape_nchw)
        return x

    x = torch.rand(*shape_nlc)
    output = nlc2nchw2nlc(test_func3, x, shape_nchw[2:], contiguous=True)
    assert output.shape == torch.Size(shape_nlc)
    assert output.is_contiguous()


def test_nchw_to_nlc():
    # Test nchw_to_nlc function
    shape_nchw = (4, 2, 5, 5)
    shape_nlc = (4, 25, 2)
    x = torch.rand(*shape_nchw)
    y = nchw_to_nlc(x)
    assert y.shape == torch.Size(shape_nlc)


def test_nlc_to_nchw():
    # Test nlc_to_nchw function
    shape_nchw = (4, 2, 5, 5)
    shape_nlc = (4, 25, 2)
    x = torch.rand(*shape_nlc)
    y = nlc_to_nchw(x, (5, 5))
    assert y.shape == torch.Size(shape_nchw)
