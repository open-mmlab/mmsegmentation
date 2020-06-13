import numpy as np
import pytest
import torch

from mmseg.models.losses import reduce_loss, weight_reduce_loss


def test_utils():
    loss = torch.rand(1, 3, 4, 4)
    weight = torch.zeros(1, 3, 4, 4)
    weight[:, :, :2, :2] = 1

    # test reduce_loss()
    reduced = reduce_loss(loss, 'none')
    assert reduced is loss

    reduced = reduce_loss(loss, 'mean')
    np.testing.assert_almost_equal(reduced.numpy(), loss.mean())

    reduced = reduce_loss(loss, 'sum')
    np.testing.assert_almost_equal(reduced.numpy(), loss.sum())

    # test weight_reduce_loss()
    reduced = weight_reduce_loss(loss, weight=None, reduction='none')
    assert reduced is loss

    reduced = weight_reduce_loss(loss, weight=weight, reduction='mean')
    target = (loss * weight).mean()
    np.testing.assert_almost_equal(reduced.numpy(), target)

    reduced = weight_reduce_loss(loss, weight=weight, reduction='sum')
    np.testing.assert_almost_equal(reduced.numpy(), (loss * weight).sum())

    with pytest.raises(AssertionError):
        weight_wrong = weight[0, 0, ...]
        _ = weight_reduce_loss(loss, weight=weight_wrong, reduction='mean')

    with pytest.raises(AssertionError):
        weight_wrong = weight[:, 0:2, ...]
        _ = weight_reduce_loss(loss, weight=weight_wrong, reduction='mean')


def test_ce_loss():
    from mmseg.models import build_loss

    # use_mask and use_sigmoid cannot be true at the same time
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='CrossEntropyLoss',
            use_mask=True,
            use_sigmoid=True,
            loss_weight=1.0)
        build_loss(loss_cfg)

    # test loss with class weights
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=[0.8, 0.2],
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    fake_pred = torch.Tensor([[100, -100]])
    fake_label = torch.Tensor([1]).long()
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))

    loss_cls_cfg = dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(200.))

    loss_cls_cfg = dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(0.))

    # TODO test use_mask
