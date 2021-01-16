import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def calculate_weights(relaxed_one_hot, norm=False, upper_bound=1.0):
    """Calculate image based classes' weights."""
    assert relaxed_one_hot.dim() == 4
    # [num_classes+1, b, h, w]
    relaxed_one_hot = relaxed_one_hot.transpose(0, 1)
    relaxed_one_hot = relaxed_one_hot[:-1, relaxed_one_hot[-1] != 1]
    hist = relaxed_one_hot.sum(dim=1)
    hist_norm = hist / hist.sum()
    if norm:
        weights = ((hist != 0) * upper_bound *
                   (1 / (hist_norm + torch.finfo(hist_norm.dtype).tiny))) + 1
    else:
        weights = ((hist != 0) * upper_bound * (1 - hist_norm)) + 1
    return weights


def soft_cross_entropy(pred,
                       relaxed_one_hot,
                       class_weight=None,
                       customsoftmax=False):
    """The function for SoftCrossEntropyLoss.

    class_weight is a manual rescaling weight given to each class. If given,
    the size of tensor eqquals to the number of class.
    """

    pred = F.softmax(pred, dim=1)
    # avoid 0
    pred += torch.finfo(pred.dtype).tiny
    if customsoftmax:
        pred = torch.log(
            torch.max(pred, (relaxed_one_hot *
                             (pred * relaxed_one_hot).sum(1, keepdim=True))))
    else:
        pred = torch.log(pred)
    loss = -1 * relaxed_one_hot * pred
    if class_weight is not None:
        loss *= class_weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return loss.sum(dim=1)


@LOSSES.register_module()
class SoftCrossEntropyLoss(nn.Module):
    """SoftCrossEntropyLoss.

    Args:
        img_based_class_weights (None | 'norm' | 'no_norm'): Whether to use
            the training images to calculate classes' weights. Default: None.
            'norm' and 'no_norm' are two methods to calculate classes; weights.
        batch_weights (bool): Calculate calsses' weights with batch images or
            image-wise.
        upper_bound (float): The upper bound of classes' weights to add.
        customsoftmax (bool): Whether to use customsoftmax or softmax.
            Default: False.
        reduction (str, optional): . Defaults: 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults: None.
        loss_weight (float, optional): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 img_based_class_weights=None,
                 batch_weights=True,
                 upper_bound=1.0,
                 customsoftmax=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(SoftCrossEntropyLoss, self).__init__()
        self.img_based_class_weights = img_based_class_weights
        self.batch_weights = batch_weights
        self.upper_bound = upper_bound
        self.customsoftmax = customsoftmax
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                relaxed_one_hot,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        The shape of cls_score is [b, num_class, h, w]. The shape of
        relaxed_one_hot is [b, num_class+1, h, w]. relaxed_one_hot[:,
        num_class+1, :, :] encodes the ignore_index, if the value is 1, it is
        ignored and does not contribute to the input gradient.
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        assert cls_score.dim() == relaxed_one_hot.dim()
        assert cls_score.shape[1] + 1 == relaxed_one_hot.shape[1]
        relaxed_one_hot = relaxed_one_hot.float()
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        if (class_weight is None) and (self.img_based_class_weights is not None
                                       ) and (not self.batch_weights):
            b, _, h, w = cls_score.shape
            loss = cls_score.new_zeros(size=(b, h, w))
            for i in range(b):
                class_weight = calculate_weights(
                    relaxed_one_hot=relaxed_one_hot[i].unsqueeze(0),
                    norm=self.img_based_class_weights == 'norm',
                    upper_bound=self.upper_bound)
                loss[i] = soft_cross_entropy(
                    cls_score[i].unsqueeze(0),
                    relaxed_one_hot[i, :-1].unsqueeze(0),
                    class_weight=class_weight,
                    customsoftmax=self.customsoftmax)
        else:
            if (class_weight is None) and (self.img_based_class_weights
                                           is not None) and self.batch_weights:
                class_weight = calculate_weights(
                    relaxed_one_hot=relaxed_one_hot,
                    norm=self.img_based_class_weights == 'norm',
                    upper_bound=self.upper_bound)
            loss = soft_cross_entropy(
                cls_score,
                relaxed_one_hot[:, :-1],
                class_weight=class_weight,
                customsoftmax=self.customsoftmax)
        # norm and ignore
        loss *= 1 / relaxed_one_hot[:, :-1].sum(
            dim=1) * (1 - relaxed_one_hot[:, -1])

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss
