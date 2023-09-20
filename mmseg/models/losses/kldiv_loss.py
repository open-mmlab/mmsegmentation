# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class KLDivLoss(nn.Module):

    def __init__(self,
                 temperature: float = 1.0,
                 reduction: str = 'mean',
                 loss_name: str = 'loss_kld'):
        """Kullback-Leibler divergence Loss.

        <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>

        Args:
            temperature (float, optional): Temperature param
            reduction  (str,  optional): The method to reduce the loss into a
            scalar. Default is "mean". Options are "none", "sum",
            and "mean"
        """

        assert isinstance(temperature, (float, int)), \
            'Expected temperature to be' \
            f'float or int, but got {temperature.__class__.__name__} instead'
        assert temperature != 0., 'Temperature must not be zero'

        assert reduction in ['mean', 'none', 'sum'], \
            'Reduction must be one of the options ("mean", ' \
            f'"sum", "none"), but got {reduction}'

        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self._loss_name = loss_name

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Forward function. Calculate KL divergence Loss.

        Args:
            input (Tensor): Logit tensor,
                the data type is float32 or float64.
                The shape is (N, C) where N is batchsize and C  is number of
                channels.
                If there more than 2 dimensions, shape is (N, C, D1, D2, ...
                Dk), k>= 1
            target (Tensor): Logit tensor,
                the data type is float32 or float64.
                input and target must be with the same shape.

        Returns:
            (Tensor): Reduced loss.
        """
        assert isinstance(input, torch.Tensor), 'Expected input to' \
            f'be Tensor, but got {input.__class__.__name__} instead'
        assert isinstance(target, torch.Tensor), 'Expected target to' \
            f'be Tensor, but got {target.__class__.__name__} instead'

        assert input.shape == target.shape, 'Input and target ' \
            'must have same shape,' \
            f'but got shapes {input.shape} and {target.shape}'

        input = F.softmax(input / self.temperature, dim=1)
        target = F.softmax(target / self.temperature, dim=1)

        loss = F.kl_div(input, target, reduction='none', log_target=False)
        loss = loss * self.temperature**2

        batch_size = input.shape[0]

        if self.reduction == 'sum':
            # Change view to calculate instance-wise sum
            loss = loss.view(batch_size, -1)
            return torch.sum(loss, dim=1)

        elif self.reduction == 'mean':
            # Change view to calculate instance-wise mean
            loss = loss.view(batch_size, -1)
            return torch.mean(loss, dim=1)

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
