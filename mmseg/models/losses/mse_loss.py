import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@ LOSSES.register_module
class MSELoss(nn.Module):

    def __init__(self, reduction="mean", loss_weight=1.0, avg_non_ignore=True, loss_name='loss_mse'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        self.loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        target_expanded = target.data.unsqueeze(1).clone()
        mask_ignore = (target_expanded == 255)
        target_expanded[mask_ignore] = 0
        one_hot_gt = torch.zeros_like(pred, dtype=torch.uint8).scatter_(1, target_expanded, 1)
        probs = F.softmax(pred, dim=1)

        loss = torch.sum((one_hot_gt - probs)**2, dim=1, keepdim=True)
        if ignore_index:
            loss = torch.where(mask_ignore, torch.zeros_like(loss), loss)

        avg_factor = target.numel() - (target == ignore_index).sum().item()
        if reduction == 'mean':
            loss_cls = loss.sum() / avg_factor
        elif reduction == 'sum':
            loss_cls = loss.sum()
        else:
            loss_cls = loss

        return self.loss_weight * loss_cls
