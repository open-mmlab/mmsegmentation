# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.losses.kldiv_loss import KLDivLoss


def test_kldiv_loss_with_none_reduction():
    loss_class = KLDivLoss
    pred = torch.rand((8, 5, 5))
    target = torch.rand((8, 5, 5))
    reduction = 'none'

    # Test loss forward
    loss = loss_class(reduction=reduction)(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (8, 5, 5), f'{loss.shape}'


def test_kldiv_loss_with_mean_reduction():
    loss_class = KLDivLoss
    pred = torch.rand((8, 5, 5))
    target = torch.rand((8, 5, 5))
    reduction = 'mean'

    # Test loss forward
    loss = loss_class(reduction=reduction)(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (8, ), f'{loss.shape}'


def test_kldiv_loss_with_sum_reduction():
    loss_class = KLDivLoss
    pred = torch.rand((8, 5, 5))
    target = torch.rand((8, 5, 5))
    reduction = 'sum'

    # Test loss forward
    loss = loss_class(reduction=reduction)(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (8, ), f'{loss.shape}'
