# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES
from .cross_entropy_loss import _expand_onehot_labels
from .utils import reduce_loss


def phi_loss(inputs, targets, gamma, smooth, weight=None, reduction='mean'):

    inputs = inputs.sigmoid()
    targets = targets.type_as(inputs)

    # flatten label and prediction tensors
    inputs = inputs.reshape(inputs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    weight = weight.reshape(weight.size(0), -1)

    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()
    TN = inputs[targets == 0.].sum()

    phi = ((TP * TN - FP * FN) + smooth) / torch.sqrt((TP + FP) * (TP + FN) *
                                                      (TN + FP) *
                                                      (TN + FN) + smooth)
    loss = (1. - phi)**gamma  # focal_phi_loss
    loss = reduce_loss(loss, reduction=reduction)
    return loss


@LOSSES.register_module()
class PhiLoss(nn.Module):
    """PhiLoss.

    This loss is proposed in `Novel Focal Phi Loss for Power Line
    Segmentation with Auxiliary
    Classifier U-Net <https://doi.org/10.3390/s21082803>.

    Args:
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        smooth (float, optional):  A float number to smooth loss.
            Default: 1.0
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 gamma=1.0,
                 reduction='mean',
                 smooth=1,
                 loss_weight=1.0,
                 loss_name='phi_loss'):
        super(PhiLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth = smooth
        self._loss_name = loss_name

    def forward(self,
                inputs,
                targets,
                weight=None,
                label_weights=None,
                reduction_override=None,
                ignore_index=-100):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if inputs.dim() != targets.dim():
            assert (inputs.dim() == 2 and targets.dim() == 1) or (
                inputs.dim() == 4 and targets.dim() == 3), \
                'Only pred shape [N, C], label shape [N] or pred' \
                'shape [N, C, H, W], label shape [N, H, W] are supported'
            # `weight` returned from `_expand_onehot_labels`
            # has been treated for valid (non-ignore) pixels
            label, label_weights, _ = _expand_onehot_labels(
                targets, label_weights, inputs.shape, ignore_index)
        phi_loss_val = phi_loss(
            inputs,
            label,
            self.gamma,
            self.smooth,
            weight,
            reduction=self.reduction)
        loss = self.loss_weight * phi_loss_val
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
