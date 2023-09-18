# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmseg.registry import MODELS
from .utils import weight_reduce_loss


def silog_loss(pred: Tensor,
               target: Tensor,
               weight: Optional[Tensor] = None,
               eps: float = 1e-4,
               reduction: Union[str, None] = 'mean',
               avg_factor: Optional[int] = None) -> Tensor:
    """Computes the Scale-Invariant Logarithmic (SI-Log) loss between
    prediction and target.

    Args:
        pred (Tensor): Predicted output.
        target (Tensor): Ground truth.
        weight (Optional[Tensor]): Optional weight to apply on the loss.
        eps (float): Epsilon value to avoid division and log(0).
        reduction (Union[str, None]): Specifies the reduction to apply to the
            output: 'mean', 'sum' or None.
        avg_factor (Optional[int]): Optional average factor for the loss.

    Returns:
        Tensor: The calculated SI-Log loss.
    """
    pred, target = pred.flatten(1), target.flatten(1)
    valid_mask = (target > eps).detach().float()

    diff_log = torch.log(target.clamp(min=eps)) - torch.log(
        pred.clamp(min=eps))

    valid_mask = (target > eps).detach() & (~torch.isnan(diff_log))
    diff_log[~valid_mask] = 0.0
    valid_mask = valid_mask.float()

    diff_log_sq_mean = (diff_log.pow(2) * valid_mask).sum(
        dim=1) / valid_mask.sum(dim=1).clamp(min=eps)
    diff_log_mean = (diff_log * valid_mask).sum(dim=1) / valid_mask.sum(
        dim=1).clamp(min=eps)

    loss = torch.sqrt(diff_log_sq_mean - 0.5 * diff_log_mean.pow(2))

    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class SiLogLoss(nn.Module):
    """Compute SiLog loss.

    Args:
        reduction (str, optional): The method used
            to reduce the loss. Options are "none",
            "mean" and "sum". Defaults to 'mean'.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        eps (float): Avoid dividing by zero. Defaults to 1e-3.
        loss_name (str, optional): Name of the loss item. If you want this
            loss item to be included into the backward graph, `loss_` must
            be the prefix of the name. Defaults to 'loss_silog'.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-6,
                 loss_name='loss_silog'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):

        assert pred.shape == target.shape, 'the shapes of pred ' \
            f'({pred.shape}) and target ({target.shape}) are mismatch'

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * silog_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
        )

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
