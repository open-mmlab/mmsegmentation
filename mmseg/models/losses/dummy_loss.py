import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def dummy_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


@LOSSES.register_module
class DummyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DummyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=1.,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * dummy_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
