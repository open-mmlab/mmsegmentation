# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.losses import DiceLoss


@pytest.mark.parametrize('naive_dice', [True, False])
def test_dice_loss(naive_dice):
    loss_class = DiceLoss
    pred = torch.rand((10, 4, 4))
    target = torch.rand((10, 4, 4))
    weight = torch.rand(10)

    # Test loss forward
    loss = loss_class(naive_dice=naive_dice)(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with weight
    loss = loss_class(naive_dice=naive_dice)(pred, target, weight)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with reduction_override
    loss = loss_class(naive_dice=naive_dice)(
        pred, target, reduction_override='mean')
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = loss_class(naive_dice=naive_dice)(pred, target, avg_factor=10)
    assert isinstance(loss, torch.Tensor)

    with pytest.raises(ValueError):
        # loss can evaluate with avg_factor only if
        # reduction is None, 'none' or 'mean'.
        reduction_override = 'sum'
        loss_class(naive_dice=naive_dice)(
            pred, target, avg_factor=10, reduction_override=reduction_override)

    # Test loss forward with avg_factor and reduction
    for reduction_override in [None, 'none', 'mean']:
        loss_class(naive_dice=naive_dice)(
            pred, target, avg_factor=10, reduction_override=reduction_override)
        assert isinstance(loss, torch.Tensor)

    # Test loss forward with has_acted=False and use_sigmoid=False
    with pytest.raises(NotImplementedError):
        loss_class(
            use_sigmoid=False, activate=True, naive_dice=naive_dice)(pred,
                                                                     target)

    # Test loss forward with weight.ndim != loss.ndim
    with pytest.raises(AssertionError):
        weight = torch.rand((2, 8))
        loss_class(naive_dice=naive_dice)(pred, target, weight)

    # Test loss forward with len(weight) != len(pred)
    with pytest.raises(AssertionError):
        weight = torch.rand(8)
        loss_class(naive_dice=naive_dice)(pred, target, weight)
