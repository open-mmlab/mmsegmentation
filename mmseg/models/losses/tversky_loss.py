"""Modified from
https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
(Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


@weighted_loss
def tversky_loss(pred,
                 target,
                 valid_mask,
                 smooth=1,
                 exponent=2,
                 class_weight=None,
                 ignore_index=255,
                 alpha=0.3,
                 beta=0.7):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            tversky_loss = binary_tversky_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent,
                alpha=alpha,
                beta=beta)
            if class_weight is not None:
                tversky_loss *= class_weight[i]
            total_loss += tversky_loss
    return total_loss / num_classes


@weighted_loss
def binary_tversky_loss(pred,
                        target,
                        valid_mask,
                        smooth=1,
                        exponent=2,
                        alpha=0.3,
                        beta=0.7,
                        **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    TP = torch.sum(torch.mul(pred, target) * valid_mask, dim=1)
    FP = torch.sum(torch.mul(pred, 1 - target) * valid_mask, dim=1)
    FN = torch.sum(torch.mul(1 - pred, target) * valid_mask, dim=1)
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - tversky


@LOSSES.register_module()
class TverskyLoss(nn.Module):
    """TverskyLoss.

    This loss is proposed in `Tversky loss function for image segmentation
    using 3D fully convolutional deep networks
    <https://arxiv.org/abs/1706.05721>`

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        alpha(float, in [0, 1]):
            The coefficient of false positives. Default: 0.3
        beta (float, in [0, 1]):
            The coefficient of false negatives. Default: 0.7
            Note: alpha + beta = 1
    """

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 alpha=0.3,
                 beta=0.7,
                 **kwards):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * tversky_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index,
            alpha=self.alpha,
            beta=self.beta)
        return loss
