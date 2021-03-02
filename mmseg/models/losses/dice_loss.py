import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def dice_loss(pred,
              target,
              valid_mask,
              smooth=1, 
              exponent=2,
              class_weight=None,
              ignore_index=-1):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(pred[:, i], target[..., i], valid_mask=valid_mask, smooth=smooth, exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes

@weighted_loss
def binary_dice_loss(pred, 
                     target,
                     valid_mask,
                     smooth=1, 
                     exponent=2,
                     **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((pred.pow(exponent) + target.pow(exponent)) * valid_mask, dim=1) + smooth

    return 1 - num / den

@LOSSES.register_module()
class DiceLoss(nn.Module):
    """DiceLoss.

    """
    def __init__(self, 
                 loss_type='multi_class',
                 smooth=1, 
                 exponent=2,
                 reduction='mean',
                 class_weight=None, 
                 loss_weight=1.0,
                 ignore_index=-1):
        super(DiceLoss, self).__init__()
        assert loss_type in ['multi_class', 'binary']
        if loss_type == 'multi_class':
            self.cls_criterion = dice_loss
        else:
            self.cls_criterion = binary_dice_loss
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, 
                pred, 
                target, 
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None      
        
        pred = F.softmax(pred, dim=1)
        one_hot_target =  F.one_hot(torch.clamp_min(target.long(), 0))
        valid_mask = (target != self.ignore_index).long()
        
        loss = self.loss_weight * self.cls_criterion(
            pred, 
            one_hot_target, 
            valid_mask=valid_mask,
            reduction=reduction, 
            avg_factor=avg_factor, 
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss
