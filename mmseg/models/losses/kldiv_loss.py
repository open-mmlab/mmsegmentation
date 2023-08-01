import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class KLDivLoss(nn.Module):
    def __init__(self,
                 temperature=1.0,
                 reduction='mean'):
        """Kullback-Leibler divergence Loss
        <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>

        Args:
            temperature (float, optional): Temperature param
            reduction  (str,  optional): The method to reduce the loss into a
            scalar. Default is "mean". Options are "none", "sum",
            and "mean"
        """

        assert isinstance(temperature, (float, int)), "Epected temperature to be" \
            f"float or int, but got {temperature.__class__.__name__} instead"
        assert temperature != 0., "Temperature must not be zero"

        assert reduction in ["mean", "none", "sum"], \
            "Reduction must be one of the options ('mean', " \
            f"'sum', 'none'), but got {reduction}"

        super(KLDivLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, input, target):
        """
        Forward function. Calculate KL divergence Loss.

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
        assert isinstance(input, torch.Tensor), "Expected input to" \
            f"be Tensor, but got {input.__class__.__name__} instead"
        assert isinstance(target, torch.Tensor), "Expected target to" \
            f"be Tensor, but got {target.__class__.__name__} instead"

        assert input.shape == target.shape, "Input and target " \
            "must have same shape," \
            f"but got shapes {input.shape} and {target.shape}"

        input = F.softmax(input / self.temperature, dim=1)
        target = F.softmax(target / self.temperature, dim=1)

        loss = F.kl_div(input, target, reduction='none', log_target=False)
        loss = loss * self.temperature * self.temperature

        batch_size = input.shape[0]


        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            # Change view to calculate instance-wise sum
            loss = loss.view(batch_size, -1)
            return torch.sum(loss, dim=1)

        elif self.reduction == 'mean':
            # Change view to calculate instance-wise mean
            loss = loss.view(batch_size, -1)
            return torch.mean(loss, dim=1)

        return loss

