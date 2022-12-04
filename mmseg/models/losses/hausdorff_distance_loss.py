# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from
https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_HD.py#L106
(Apache-2.0 license)"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


def compute_dtm(mask_arr, output_shape, normalized=False):
    """compute the distance transform map of foreground in binary mask.

    Args:
        mask_arr (array): segmentation mask, shape=(batch_size, channel, x, y)
        output_shape (tuple): predict mask shape after F.Softmax(dim=1),
                              shape=(batch_size, 2, channel, x, y)
        normalized (bool): whether compute the dtm by normalized

    Returns:
        fg_dtm (array)：the foreground Distance Map (SDM), shape=out_shape
            dtm(x) = 0;     x out of segmentation or x in segmentation boundary
                   = inf|x - y|;    x in segmentation
                                    if normalized is True, dtm(x) to [0, 1]
    """
    fg_dtm = np.zeros(output_shape)

    for b in range(output_shape[0]):
        for c in range(1, output_shape[1]):
            positive_mask = mask_arr[b].astype(bool)
            if positive_mask.any():
                positive_distance = distance_transform_edt(positive_mask)
                fg_dtm[b][c] = positive_distance/np.max(positive_distance) \
                    if normalized else positive_mask
    return fg_dtm


@weighted_loss
def binary_hausdorff_distance_loss(pred, target, valid_mask):
    assert pred.shape[0] == target.shape[0]

    with torch.no_grad:
        gt_dtm_npy = compute_dtm((target * valid_mask).cpu.numpy(), pred.shape)
        gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(pred.device.index)
        pred_dtm_npy = compute_dtm(
            (pred * valid_mask)[:, 1, ...].cpu.numpy() > 0.5, pred.shape)
        pred_dtm = torch.from_numpy(pred_dtm_npy).float().cuda(
            pred.device.index)

    # compute hausdorff distance loss for binary segmentation
    delta_s = (pred[:, 1, ...] - target.float())**2
    dtm = pred_dtm[:, 1, ...]**2 + gt_dtm[:, 1, ...]**2
    hd_loss = torch.einsum('bcxy,bcxy->bcxy', delta_s, dtm).mean()

    return hd_loss


@weighted_loss
def hausdorff_distance_loss(pred,
                            target,
                            valid_mask,
                            class_weight=None,
                            ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            hd_loss = binary_hausdorff_distance_loss(
                pred[:, i], target[..., i], valid_mask=valid_mask)
            if class_weight is not None:
                hd_loss *= class_weight[i]
            total_loss += hd_loss
    return total_loss / num_classes


@LOSSES.register_module()
class HausdorffDistanceLoss(nn.Module):
    """HausdorffDistanceLoss. This loss is proposed in `the Hausdorff Distance
    in Medical Image Segmentation with Convolutional Neural Networks.

    <https://arxiv.org/abs/1904.10030>`_.
    Args:
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_hausdorff_distance'.
    """

    def __init__(self,
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_hausdorff_distance'):
        super().__init__()
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * hausdorff_distance_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
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
