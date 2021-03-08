import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import expand_onehot_labels, weight_reduce_loss


def focal_loss(pred,
               label,
               weight=None,
               class_weight=None,
               reduction='mean',
               avg_factor=None,
               ignore_index=-100,
               eps=1e-7,
               gamma=0):
    B, C, H, W = pred.size()

    if label.ndim == 3:
        label = label.unsqueeze(dim=3)
    if weight is None:
        weight = torch.ones_like(pred)
    if class_weight is not None:
        weight *= torch.tensor(class_weight).reshape(
            1, C, 1, 1).expand_as(weight).to(weight)

    if ignore_index is not None:
        mask = (label != ignore_index)
        mask = mask.reshape(mask.shape[0], 1, mask.shape[1],
                            mask.shape[2]).expand_as(weight)

    target = expand_onehot_labels(label, C)

    probs = F.softmax(input, dim=1)
    probs = (probs * target)
    probs = probs.clamp(eps, 1. - eps)

    log_p = probs.log()
    # print('probs size= {}'.format(probs.size()))
    # print(probs)

    loss = -(torch.pow((1 - probs), gamma)) * log_p
    # print('-----bacth_loss------')
    # print(batch_loss)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 loss_weight=1,
                 gamma=0,
                 reduction='mean',
                 class_weight=None,
                 one_hot=True,
                 eps=1e-7,
                 **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.class_weight = class_weight
        self.one_hot = one_hot
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.cls_criterion = focal_loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """only support ignore at 0."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            eps=self.eps,
            gamma=self.gamma,
            **kwargs)
        return loss_cls
