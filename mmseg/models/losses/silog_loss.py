# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from mmseg.registry import MODELS
from .utils import weight_reduce_loss


def silog_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Union[Tensor, None],
    lambd: float = 0.5,
    eps: float = 1e-6,
    reduction: Union[str, None] = 'mean',
    avg_factor: Union[int, None] = None,
):
    pred, target = pred.flatten(1), target.flatten(1)
    valid_mask = (target > 0).detach().float()
    diff_log = torch.log(target.clip(min=eps)) - torch.log(pred.clip(min=eps))
    diff_log_sq_mean = diff_log.pow(2).sum(dim=1) / valid_mask.sum(dim=1)
    diff_log_mean = diff_log.sum(dim=1) / valid_mask.sum(dim=1)
    loss = torch.sqrt(diff_log_sq_mean - lambd * diff_log_mean.pow(2))

    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)

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
                 lambd=0.5,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-6,
                 loss_name='loss_silog'):
        super().__init__()
        self.lambd = lambd
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
            lambd=self.lambd,
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
