# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.utils import nchw2nlc2nchw, nlc2nchw2nlc


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
