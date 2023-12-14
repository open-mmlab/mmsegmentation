# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmseg.models.losses import CrossEntropyLoss, weight_reduce_loss


def test_cross_entropy_loss_class_weights():
    loss_class = CrossEntropyLoss
    pred = torch.rand((1, 10, 4, 4))
    target = torch.randint(0, 10, (1, 4, 4))
    class_weight = torch.ones(10)
    avg_factor = target.numel()

    cross_entropy_loss = F.cross_entropy(
        pred, target, weight=class_weight, reduction='none', ignore_index=-100)

    expected_loss = weight_reduce_loss(
        cross_entropy_loss,
        weight=None,
        reduction='mean',
        avg_factor=avg_factor)

    # Test loss forward
    loss = loss_class(class_weight=class_weight.tolist())(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert expected_loss == loss
