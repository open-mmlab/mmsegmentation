import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def calculate_weights(label, num_classes, norm=False, upper_bound=1.0):
    """Calculate image based classes' weights."""
    hist = label.float().histc(bins=num_classes, min=0, max=num_classes - 1)
    hist_norm = hist / hist.sum()
    if norm:
        weights = ((hist != 0) * upper_bound * (1 / (hist_norm + 1e-6))) + 1
    else:
        weights = ((hist != 0) * upper_bound * (1 - hist_norm)) + 1
    return weights


def cross_entropy(pred,
                  label,
                  weight=None,
                  img_based_class_weights=None,
                  batch_weights=True,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    if (class_weight is None) and (img_based_class_weights
                                   is not None) and (not batch_weights):
        assert pred.dim() > 2 and label.dim() > 1
        loss = torch.zeros_like(label).float()
        for i in range(pred.shape[0]):
            class_weight = calculate_weights(
                label=label[i],
                num_classes=pred.shape[1],
                norm=img_based_class_weights == 'norm')
            loss[i] = F.cross_entropy(
                pred[i].unsqueeze(0),
                label[i].unsqueeze(0),
                weight=class_weight,
                reduction='none',
                ignore_index=ignore_index)
    else:
        if (class_weight is None) and (img_based_class_weights
                                       is not None) and batch_weights:
            class_weight = calculate_weights(
                label=label,
                num_classes=pred.shape[1],
                norm=img_based_class_weights == 'norm')
            # print(class_weight)
        loss = F.cross_entropy(
            pred,
            label,
            weight=class_weight,
            reduction='none',
            ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        img_based_class_weights (None | 'norm' | 'no_norm'): Whether to use
            the training images to calculate classes' weights. Default is None.
            'norm' and 'no_norm' are two methods to calculate classes; weights.
        batch_weights (bool): Calculate calsses' weights with batch images or
            image-wise.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 img_based_class_weights=None,
                 batch_weights=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.img_based_class_weights = img_based_class_weights
        self.batch_weights = batch_weights
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        if (not self.use_sigmoid) and (not self.use_mask):
            loss_cls = self.loss_weight * self.cls_criterion(
                cls_score,
                label,
                weight,
                img_based_class_weights=self.img_based_class_weights,
                batch_weights=self.batch_weights,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
        else:
            loss_cls = self.loss_weight * self.cls_criterion(
                cls_score,
                label,
                weight,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
        return loss_cls
