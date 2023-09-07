# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import TASK_UTILS


class BaseMatchCost:
    """Base match cost class.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self, pred_instances: InstanceData,
                 gt_instances: InstanceData, **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (InstanceData): Instances of model predictions.
            It often includes "labels" and "scores".
            gt_instances (InstanceData): Ground truth of instance
            annotations. It usually includes "labels".

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pass


@TASK_UTILS.register_module()
class ClassificationCost(BaseMatchCost):
    """ClsSoftmaxCost.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmseg.models.assigners import ClassificationCost
        >>> import torch
        >>> self = ClassificationCost()
        >>> cls_pred = torch.rand(4, 3)
        >>> gt_labels = torch.tensor([0, 1, 2])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(cls_pred, gt_labels)
        tensor([[-0.3430, -0.3525, -0.3045],
            [-0.3077, -0.2931, -0.3992],
            [-0.3664, -0.3455, -0.2881],
            [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight: Union[float, int] = 1) -> None:
        super().__init__(weight=weight)

    def __call__(self, pred_instances: InstanceData,
                 gt_instances: InstanceData, **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (InstanceData): "scores" inside is
                predicted classification logits, of shape
                (num_queries, num_class).
            gt_instances (InstanceData): "labels" inside should have
                shape (num_gt, ).

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        assert hasattr(pred_instances, 'scores'), \
            "pred_instances must contain 'scores'"
        assert hasattr(gt_instances, 'labels'), \
            "gt_instances must contain 'labels'"
        pred_scores = pred_instances.scores
        gt_labels = gt_instances.labels

        pred_scores = pred_scores.softmax(-1)
        cls_cost = -pred_scores[:, gt_labels]

        return cls_cost * self.weight


@TASK_UTILS.register_module()
class DiceCost(BaseMatchCost):
    """Cost of mask assignments based on dice losses.

    Args:
        pred_act (bool): Whether to apply sigmoid to mask_pred.
            Defaults to False.
        eps (float): Defaults to 1e-3.
        naive_dice (bool): If True, use the naive dice loss
            in which the power of the number in the denominator is
            the first power. If False, use the second power that
            is adopted by K-Net and SOLO. Defaults to True.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 pred_act: bool = False,
                 eps: float = 1e-3,
                 naive_dice: bool = True,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.pred_act = pred_act
        self.eps = eps
        self.naive_dice = naive_dice

    def _binary_mask_dice_loss(self, mask_preds: Tensor,
                               gt_masks: Tensor) -> Tensor:
        """
        Args:
            mask_preds (Tensor): Mask prediction in shape (num_queries, *).
            gt_masks (Tensor): Ground truth in shape (num_gt, *)
                store 0 or 1, 0 for negative class and 1 for
                positive class.

        Returns:
            Tensor: Dice cost matrix in shape (num_queries, num_gt).
        """
        mask_preds = mask_preds.flatten(1)
        gt_masks = gt_masks.flatten(1).float()
        numerator = 2 * torch.einsum('nc,mc->nm', mask_preds, gt_masks)
        if self.naive_dice:
            denominator = mask_preds.sum(-1)[:, None] + \
                gt_masks.sum(-1)[None, :]
        else:
            denominator = mask_preds.pow(2).sum(1)[:, None] + \
                gt_masks.pow(2).sum(1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self, pred_instances: InstanceData,
                 gt_instances: InstanceData, **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (InstanceData): Predicted instances which
                must contain "masks".
            gt_instances (InstanceData): Ground truth which must contain
                "mask".

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        assert hasattr(pred_instances, 'masks'), \
            "pred_instances must contain 'masks'"
        assert hasattr(gt_instances, 'masks'), \
            "gt_instances must contain 'masks'"
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks

        if self.pred_act:
            pred_masks = pred_masks.sigmoid()
        dice_cost = self._binary_mask_dice_loss(pred_masks, gt_masks)
        return dice_cost * self.weight


@TASK_UTILS.register_module()
class CrossEntropyLossCost(BaseMatchCost):
    """CrossEntropyLossCost.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
                of softmax. Defaults to True.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 use_sigmoid: bool = True,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred: Tensor,
                              gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_queries, 1, *) or
                (num_queries, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).

        Returns:
            Tensor: Cross entropy cost matrix in shape (num_queries, num_gt).
        """
        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
            torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost

    def __call__(self, pred_instances: InstanceData,
                 gt_instances: InstanceData, **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``masks``.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        assert hasattr(pred_instances, 'masks'), \
            "pred_instances must contain 'masks'"
        assert hasattr(gt_instances, 'masks'), \
            "gt_instances must contain 'masks'"
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(pred_masks, gt_masks)
        else:
            raise NotImplementedError

        return cls_cost * self.weight
