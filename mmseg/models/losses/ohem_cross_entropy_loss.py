# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class OhemCrossEntropy(nn.Module):
    """OhemCrossEntropy loss.

    This func is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L43>`_.  # noqa

    Licensed under the MIT License.

    Args:
        ignore_label (int): Labels to ignore when computing the loss.
            Default: 255
        thresh (float, optional): The threshold for hard example selection.
            Below which, are prediction with low confidence. If not
            specified, the hard examples will be pixels of top ``min_kept``
            loss. Default: 0.7.
        min_kept (int, optional): The minimum number of predictions to keep.
            Default: 100000.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 ignore_label: int = 255,
                 thres: float = 0.7,
                 min_kept: int = 100000,
                 loss_weight: float = 1.0,
                 class_weight: Optional[Union[List[float], str]] = None,
                 loss_name: str = 'loss_ohem'):
        super().__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.class_weight = class_weight

    def forward(self, score: Tensor, target: Tensor) -> Tensor:
        """Forward function.
        Args:
            score (Tensor): Predictions of the segmentation head.
            target (Tensor): Ground truth of the image.

        Returns:
            Tensor: Loss tensor.
        """
        # score: (N, C, H, W)
        pred = F.softmax(score, dim=1)
        if self.class_weight is not None:
            class_weight = score.new_tensor(self.class_weight)
        else:
            class_weight = None

        pixel_losses = F.cross_entropy(
            score,
            target,
            weight=class_weight,
            ignore_index=self.ignore_label,
            reduction='none').contiguous().view(-1)  # (N*H*W)
        mask = target.contiguous().view(-1) != self.ignore_label  # (N*H*W)

        tmp_target = target.clone()  # (N, H, W)
        tmp_target[tmp_target == self.ignore_label] = 0
        # pred: (N, C, H, W) -> (N*H*W, C)
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        # pred: (N*H*W, C) -> (N*H*W), ind: (N*H*W)
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        if pred.numel() > 0:
            min_value = pred[min(self.min_kept, pred.numel() - 1)]
        else:
            return score.new_tensor(0.0)
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return self.loss_weight * pixel_losses.mean()

    @property
    def loss_name(self):
        return self.loss_name_
