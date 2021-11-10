# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from mmseg.models import build_loss


# test focal loss with use_sigmoid=False
def test_use_sigmoid():
    # can't init with use_sigmoid=True
    with pytest.raises(AssertionError):
        loss_cfg = dict(type='FocalLoss', use_sigmoid=False)
        build_loss(loss_cfg)

    # can't forward with use_sigmoid=True
    with pytest.raises(NotImplementedError):
        loss_cfg = dict(type='FocalLoss', use_sigmoid=True)
        focal_loss = build_loss(loss_cfg)
        focal_loss.use_sigmoid = False
        fake_pred = torch.rand(3, 2, 2, 2)
        fake_target = torch.rand(3, 2, 2)
        focal_loss(fake_pred, fake_target)


# reduction type must be 'none', 'mean' or 'sum'
def test_wrong_reduction_type():
    # can't init with wrong reduction
    with pytest.raises(AssertionError):
        loss_cfg = dict(type='FocalLoss', reduction='test')
        build_loss(loss_cfg)

    # can't forward with wrong reduction override
    with pytest.raises(AssertionError):
        loss_cfg = dict(type='FocalLoss')
        focal_loss = build_loss(loss_cfg)
        fake_pred = torch.rand(3, 2, 2, 2)
        fake_target = torch.rand(3, 2, 2)
        focal_loss(fake_pred, fake_target, reduction_override='test')


# test focal loss can handle input parameters with
# unacceptable types
def test_unacceptable_parameters():
    with pytest.raises(AssertionError):
        loss_cfg = dict(type='FocalLoss', gamma='test')
        build_loss(loss_cfg)
    with pytest.raises(AssertionError):
        loss_cfg = dict(type='FocalLoss', alpha='test')
        build_loss(loss_cfg)
    with pytest.raises(AssertionError):
        loss_cfg = dict(type='FocalLoss', loss_weight='test')
        build_loss(loss_cfg)
    with pytest.raises(AssertionError):
        loss_cfg = dict(type='FocalLoss', loss_name=123)
        build_loss(loss_cfg)


# test if focal loss can be correctly initialize
def test_init_focal_loss():
    loss_cfg = dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=3.0,
        alpha=3.0,
        reduction='sum')
    focal_loss = build_loss(loss_cfg)
    assert focal_loss.use_sigmoid is True
    assert focal_loss.gamma == 3.0
    assert focal_loss.alpha == 3.0
    assert focal_loss.reduction == 'sum'
    assert focal_loss.loss_weight == 1.0
    assert focal_loss.loss_name == 'loss_focal'


# test reduction override
def test_reduction_override():
    loss_cfg = dict(type='FocalLoss')
    focal_loss = build_loss(loss_cfg)
    fake_pred = torch.rand(3, 2, 2, 2)
    fake_target = torch.randint(0, 2, (3, 2, 2))
    focal_loss(fake_pred, fake_target, reduction_override='sum')


# test wrong pred and target shape
def test_wrong_pred_and_target_shape():
    loss_cfg = dict(type='FocalLoss')
    focal_loss = build_loss(loss_cfg)
    fake_pred = torch.rand(3, 2, 2, 2)
    fake_target = torch.rand(3, 2, 2, 3)
    with pytest.raises(AssertionError):
        focal_loss(fake_pred, fake_target)


# test forward with different shape of target
def test_forward_with_different_shape_of_target():
    loss_cfg = dict(type='FocalLoss')
    focal_loss = build_loss(loss_cfg)

    fake_pred = torch.rand(3, 2, 2, 2)
    fake_target = F.one_hot(torch.randint(0, 2, (3, 2, 2)), num_classes=2)
    focal_loss(fake_pred, fake_target)

    fake_target = torch.randint(0, 2, (3, 2, 2))
    focal_loss(fake_pred, fake_target)


# test forward with weight
def test_forward_with_weight():
    loss_cfg = dict(type='FocalLoss')
    focal_loss = build_loss(loss_cfg)
    fake_pred = torch.rand(3, 2, 2, 2)
    fake_target = torch.randint(0, 2, (3, 2, 2))
    weight = torch.rand(12, 1)
    focal_loss(fake_pred, fake_target, weight=weight)

    weight = torch.rand(12)
    focal_loss(fake_pred, fake_target, weight=weight)

    weight = torch.rand(12, 2)
    focal_loss(fake_pred, fake_target, weight=weight)


# test none reduction type
def test_none_reduction_type():
    loss_cfg = dict(type='FocalLoss', reduction='none')
    focal_loss = build_loss(loss_cfg)
    fake_pred = torch.rand(3, 2, 4, 4)
    fake_target = torch.randint(0, 2, (3, 4, 4))
    loss = focal_loss(fake_pred, fake_target)
    assert loss.shape == fake_pred.shape
