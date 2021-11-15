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
        fake_pred = torch.rand(3, 4, 5, 6)
        fake_target = torch.randint(0, 4, (3, 5, 6))
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
        fake_pred = torch.rand(3, 4, 5, 6)
        fake_target = torch.randint(0, 4, (3, 5, 6))
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
        loss_cfg = dict(type='FocalLoss', class_weight='test')
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
        class_weight=[1, 2, 3, 4],
        reduction='sum')
    focal_loss = build_loss(loss_cfg)
    assert focal_loss.use_sigmoid is True
    assert focal_loss.gamma == 3.0
    assert focal_loss.alpha == 3.0
    assert focal_loss.reduction == 'sum'
    assert focal_loss.class_weight == [1, 2, 3, 4]
    assert focal_loss.loss_weight == 1.0
    assert focal_loss.loss_name == 'loss_focal'


# test reduction override
def test_reduction_override():
    loss_cfg = dict(type='FocalLoss', reduction='mean')
    focal_loss = build_loss(loss_cfg)
    fake_pred = torch.rand(3, 4, 5, 6)
    fake_target = torch.randint(0, 4, (3, 5, 6))
    loss = focal_loss(fake_pred, fake_target, reduction_override='none')
    assert loss.shape == fake_pred.shape


# test wrong pred and target shape
def test_wrong_pred_and_target_shape():
    loss_cfg = dict(type='FocalLoss')
    focal_loss = build_loss(loss_cfg)
    fake_pred = torch.rand(3, 4, 5, 6)
    fake_target = torch.randint(0, 4, (3, 2, 2))
    fake_target = F.one_hot(fake_target, num_classes=4)
    fake_target = fake_target.permute(0, 3, 1, 2)
    with pytest.raises(AssertionError):
        focal_loss(fake_pred, fake_target)


# test forward with different shape of target
def test_forward_with_different_shape_of_target():
    loss_cfg = dict(type='FocalLoss')
    focal_loss = build_loss(loss_cfg)

    fake_pred = torch.rand(3, 4, 5, 6)
    fake_target = torch.randint(0, 4, (3, 5, 6))
    loss1 = focal_loss(fake_pred, fake_target)

    fake_target = F.one_hot(fake_target, num_classes=4)
    fake_target = fake_target.permute(0, 3, 1, 2)
    loss2 = focal_loss(fake_pred, fake_target)
    assert loss1 == loss2


# test forward with weight
def test_forward_with_weight():
    loss_cfg = dict(type='FocalLoss')
    focal_loss = build_loss(loss_cfg)
    fake_pred = torch.rand(3, 4, 5, 6)
    fake_target = torch.randint(0, 4, (3, 5, 6))
    weight = torch.rand(3 * 5 * 6, 1)
    loss1 = focal_loss(fake_pred, fake_target, weight=weight)

    weight2 = weight.view(-1)
    loss2 = focal_loss(fake_pred, fake_target, weight=weight2)

    weight3 = weight.expand(3 * 5 * 6, 4)
    loss3 = focal_loss(fake_pred, fake_target, weight=weight3)
    assert loss1 == loss2 == loss3


# test none reduction type
def test_none_reduction_type():
    loss_cfg = dict(type='FocalLoss', reduction='none')
    focal_loss = build_loss(loss_cfg)
    fake_pred = torch.rand(3, 4, 5, 6)
    fake_target = torch.randint(0, 4, (3, 5, 6))
    loss = focal_loss(fake_pred, fake_target)
    assert loss.shape == fake_pred.shape


# test the usage of class weight
def test_class_weight():
    loss_cfg_cw = dict(
        type='FocalLoss', reduction='none', class_weight=[1.0, 2.0, 3.0, 4.0])
    loss_cfg = dict(type='FocalLoss', reduction='none')
    focal_loss_cw = build_loss(loss_cfg_cw)
    focal_loss = build_loss(loss_cfg)
    fake_pred = torch.rand(3, 4, 5, 6)
    fake_target = torch.randint(0, 4, (3, 5, 6))
    loss_cw = focal_loss_cw(fake_pred, fake_target)
    loss = focal_loss(fake_pred, fake_target)
    weight = torch.tensor([1, 2, 3, 4]).view(1, 4, 1, 1)
    assert (loss * weight == loss_cw).all()


# test ignore index
def test_ignore_index():
    loss_cfg = dict(type='FocalLoss', reduction='none')
    # ignore_index within C classes
    focal_loss = build_loss(loss_cfg)
    fake_pred = torch.rand(3, 5, 5, 6)
    fake_target = torch.randint(0, 4, (3, 5, 6))
    dim1 = torch.randint(0, 3, (4, ))
    dim2 = torch.randint(0, 5, (4, ))
    dim3 = torch.randint(0, 6, (4, ))
    fake_target[dim1, dim2, dim3] = 4
    loss1 = focal_loss(fake_pred, fake_target, ignore_index=4)
    one_hot_target = F.one_hot(fake_target, num_classes=5)
    one_hot_target = one_hot_target.permute(0, 3, 1, 2)
    loss2 = focal_loss(fake_pred, one_hot_target, ignore_index=4)
    assert (loss1 == loss2).all()
    assert (loss1[dim1, :, dim2, dim3] == 0).all()
    assert (loss2[dim1, :, dim2, dim3] == 0).all()

    fake_pred = torch.rand(3, 4, 5, 6)
    fake_target = torch.randint(0, 4, (3, 5, 6))
    loss1 = focal_loss(fake_pred, fake_target, ignore_index=2)
    one_hot_target = F.one_hot(fake_target, num_classes=4)
    one_hot_target = one_hot_target.permute(0, 3, 1, 2)
    loss2 = focal_loss(fake_pred, one_hot_target, ignore_index=2)
    ignore_mask = one_hot_target == 2
    assert (loss1 == loss2).all()
    assert torch.sum(loss1 * ignore_mask) == 0
    assert torch.sum(loss2 * ignore_mask) == 0

    # ignore index is not in prediction's classes
    fake_pred = torch.rand(3, 4, 5, 6)
    fake_target = torch.randint(0, 4, (3, 5, 6))
    dim1 = torch.randint(0, 3, (4, ))
    dim2 = torch.randint(0, 5, (4, ))
    dim3 = torch.randint(0, 6, (4, ))
    fake_target[dim1, dim2, dim3] = 255
    loss1 = focal_loss(fake_pred, fake_target, ignore_index=255)
    assert (loss1[dim1, :, dim2, dim3] == 0).all()


# test list alpha
def test_alpha():
    loss_cfg = dict(type='FocalLoss')
    focal_loss = build_loss(loss_cfg)
    alpha_float = 0.4
    alpha = [0.4, 0.4, 0.4, 0.4]
    alpha2 = [0.1, 0.3, 0.2, 0.1]
    fake_pred = torch.rand(3, 4, 5, 6)
    fake_target = torch.randint(0, 4, (3, 5, 6))
    focal_loss.alpha = alpha_float
    loss1 = focal_loss(fake_pred, fake_target)
    focal_loss.alpha = alpha
    loss2 = focal_loss(fake_pred, fake_target)
    assert loss1 == loss2
    focal_loss.alpha = alpha2
    focal_loss(fake_pred, fake_target)
