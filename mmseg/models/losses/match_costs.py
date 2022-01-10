# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MATCH_COST


@MATCH_COST.register_module()
class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class MaskFocalLossCost(FocalLossCost):
    """Cost of mask assignments based on focal losses.

    Args:
        weight (int | float, optional): loss_weight.
        alpha (int | float, optional): focal_loss alpha.
        gamma (int | float, optional): focal_loss gamma.
        eps (float, optional): default 1e-12.
    """

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits
                in shape (N1, H, W), dtype=torch.float32.
            gt_labels (Tensor): Ground truth in shape (N2, H, W),
                dtype=torch.long.

        Returns:
            Tensor: classification cost matrix in shape (N1, N2).
        """
        cls_pred = cls_pred.reshape((cls_pred.shape[0], -1))
        gt_labels = gt_labels.reshape((gt_labels.shape[0], -1)).float()
        hw = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
                   torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / hw * self.weight


@MATCH_COST.register_module()
class ClassificationCost:
    """ClsSoftmaxCost.Borrow from mmdet.core.bbox.match_costs.match_cost.ClassificationCost

     Args:
         weight (int | float, optional): loss_weight

     Examples:
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

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class DiceCost:
    """Cost of mask assignments based on dice losses.

    Args:
        weight (int | float, optional): loss_weight. Defaults to 1.
        pred_act (bool, optional): Whether to apply sigmoid to mask_pred.
            Defaults to False.
        eps (float, optional): default 1e-12.
    """

    def __init__(self, weight=1., pred_act=False, eps=1e-3):
        self.weight = weight
        self.pred_act = pred_act
        self.eps = eps

    def binary_mask_dice_loss(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): Mask prediction in shape (N1, H, W).
            gt_masks (Tensor): Ground truth in shape (N2, H, W)
                store 0 or 1, 0 for negative class and 1 for
                positive class.

        Returns:
            Tensor: Dice cost matrix in shape (N1, N2).
        """
        mask_preds = mask_preds.reshape((mask_preds.shape[0], -1))
        gt_masks = gt_masks.reshape((gt_masks.shape[0], -1)).float()
        numerator = 2 * torch.einsum('nc,mc->nm', mask_preds, gt_masks)
        denominator = mask_preds.sum(-1)[:, None] + gt_masks.sum(-1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): Mask prediction logits in shape (N1, H, W).
            gt_masks (Tensor): Ground truth in shape (N2, H, W).

        Returns:
            Tensor: Dice cost matrix in shape (N1, N2).
        """
        if self.pred_act:
            mask_preds = mask_preds.sigmoid()
        dice_cost = self.binary_mask_dice_loss(mask_preds, gt_masks)
        return dice_cost * self.weight
