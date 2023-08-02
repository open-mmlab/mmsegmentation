# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.losses import DiceLoss


@pytest.mark.parametrize('naive_dice', [True, False])
def test_dice_loss(naive_dice):
    loss_class = DiceLoss
    pred = torch.rand((1, 10, 4, 4))
    target = torch.randint(0, 10, (1, 4, 4))
    weight = torch.rand(1)
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
    for use_sigmoid in [True, False]:
        loss_class(
            use_sigmoid=use_sigmoid, activate=True,
            naive_dice=naive_dice)(pred, target)
        assert isinstance(loss, torch.Tensor)

    # Test loss forward with weight.ndim != loss.ndim
    with pytest.raises(AssertionError):
        weight = torch.rand((2, 8))
        loss_class(naive_dice=naive_dice)(pred, target, weight)

    # Test loss forward with len(weight) != len(pred)
    with pytest.raises(AssertionError):
        weight = torch.rand(8)
        loss_class(naive_dice=naive_dice)(pred, target, weight)

    # Test _expand_onehot_labels_dice
    pred = torch.tensor([[[[1, 1], [1, 0]], [[0, 1], [1, 1]]]]).float()
    target = torch.tensor([[[0, 0], [0, 1]]])
    target_onehot = torch.tensor([[[[1, 1], [1, 0]], [[0, 0], [0, 1]]]])
    weight = torch.rand(1)
    loss = loss_class(naive_dice=naive_dice)(pred, target, weight)
    loss_onehot = loss_class(naive_dice=naive_dice)(pred, target_onehot,
                                                    weight)
    assert torch.equal(loss, loss_onehot)

    # Test Whether Loss is 0 when pred == target, eps == 0 and naive_dice=False
    target = torch.randint(0, 2, (1, 10, 4, 4))
    pred = target.float()
    target = target.sigmoid()
    weight = torch.rand(1)
    loss = loss_class(
        naive_dice=False, use_sigmoid=True, eps=0)(pred, target, weight)
    assert loss.item() == 0

    # Test ignore_index when ignore_index is the only class
    with pytest.raises(AssertionError):
        pred = torch.ones((1, 1, 4, 4))
        target = torch.randint(0, 1, (1, 4, 4))
        weight = torch.rand(1)
        loss = loss_class(
            naive_dice=naive_dice, use_sigmoid=False, ignore_index=0,
            eps=0)(pred, target, weight)

    # Test ignore_index with naive_dice = False
    pred = torch.tensor([[[[1, 1], [1, 0]], [[0, 1], [1, 1]]]]).float()
    target = torch.tensor([[[[1, 1], [1, 0]], [[1, 0], [0, 1]]]]).sigmoid()
    weight = torch.rand(1)
    loss = loss_class(
        naive_dice=False, use_sigmoid=True, ignore_index=1,
        eps=0)(pred, target, weight)
    assert loss.item() == 0
