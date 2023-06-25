# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.losses import HuasdorffDisstanceLoss


def test_huasdorff_distance_loss():
    loss_class = HuasdorffDisstanceLoss
    pred = torch.rand((10, 8, 6, 6))
    target = torch.rand((10, 6, 6))
    class_weight = torch.rand(8)

    # Test loss forward
    loss = loss_class()(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = loss_class()(pred, target, avg_factor=10)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor and reduction is None, 'sum' and 'mean'
    for reduction in [None, 'sum', 'mean']:
        loss = loss_class()(pred, target, avg_factor=10, reduction=reduction)
        assert isinstance(loss, torch.Tensor)

    # Test loss forward with class_weight
    with pytest.raises(AssertionError):
        loss_class(class_weight=class_weight)(pred, target)
