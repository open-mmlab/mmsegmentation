import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import get_class_count


@ LOSSES.register_module
class BalancedSoftmaxLoss(nn.Module):

    def __init__(self, class_count, reduction="mean", loss_weight=1.0, avg_non_ignore=True, pow=1, loss_name='loss_balanced_softmax'):
        super(BalancedSoftmaxLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        self.loss_name = loss_name
        self.class_count = torch.from_numpy(get_class_count(class_count)[:-1]).reshape(1, -1, 1, 1)**pow

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        device = pred.device
        exp = self.class_count.to(device) * torch.exp(pred)
        bal_sm = exp.div_(exp.sum(1, keepdim=True))
        loss_cls = F.nll_loss(bal_sm.log(), target, ignore_index=ignore_index, reduction=reduction)
        if loss_cls != loss_cls:
            import ipdb; ipdb.set_trace()
        return self.loss_weight * loss_cls
