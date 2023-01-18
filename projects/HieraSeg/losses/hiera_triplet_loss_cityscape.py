# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from .tree_triplet_loss import TreeTripletLoss

hiera_map = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 6]
hiera_index = [[0, 2], [2, 5], [5, 8], [8, 10], [10, 11], [11, 13], [13, 19]]

hiera = {
    'hiera_high': {
        'flat': [0, 2],
        'construction': [2, 5],
        'object': [5, 8],
        'nature': [8, 10],
        'sky': [10, 11],
        'human': [11, 13],
        'vehicle': [13, 19]
    }
}


def prepare_targets(targets):
    b, h, w = targets.shape
    targets_high = torch.ones(
        (b, h, w), dtype=targets.dtype, device=targets.device) * 255
    indices_high = []
    for index, high in enumerate(hiera['hiera_high'].keys()):
        indices = hiera['hiera_high'][high]
        for ii in range(indices[0], indices[1]):
            targets_high[targets == ii] = index
        indices_high.append(indices)

    return targets, targets_high, indices_high


def losses_hiera(predictions,
                 targets,
                 targets_top,
                 num_classes,
                 indices_high,
                 eps=1e-8):
    """Implementation of hiera loss.

    Args:
        predictions (torch.Tensor): seg logits produced by decode head.
        targets (torch.Tensor): The learning label of the prediction.
        targets_top (torch.Tensor): The hierarchy ground truth of the learning
            label.
        num_classes (int): Number of categories.
        indices_high (List[List[int]]): Hierarchy indices of each hierarchy.
        eps (float):Term added to the Logarithm to improve numerical stability.
    """
    b, _, h, w = predictions.shape
    predictions = torch.sigmoid(predictions.float())
    void_indices = (targets == 255)
    targets[void_indices] = 0
    targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)
    void_indices2 = (targets_top == 255)
    targets_top[void_indices2] = 0
    targets_top = F.one_hot(targets_top, num_classes=7).permute(0, 3, 1, 2)

    MCMA = predictions[:, :num_classes, :, :]
    MCMB = torch.zeros((b, 7, h, w)).to(predictions)
    for ii in range(7):
        MCMB[:, ii:ii + 1, :, :] = torch.max(
            torch.cat([
                predictions[:, indices_high[ii][0]:indices_high[ii][1], :, :],
                predictions[:, num_classes + ii:num_classes + ii + 1, :, :]
            ],
                      dim=1), 1, True)[0]

    MCLB = predictions[:, num_classes:num_classes + 7, :, :]
    MCLA = predictions[:, :num_classes, :, :].clone()
    for ii in range(7):
        for jj in range(indices_high[ii][0], indices_high[ii][1]):
            MCLA[:, jj:jj + 1, :, :] = torch.min(
                torch.cat([
                    predictions[:, jj:jj + 1, :, :], MCLB[:, ii:ii + 1, :, :]
                ],
                          dim=1), 1, True)[0]

    valid_indices = (~void_indices).unsqueeze(1)
    num_valid = valid_indices.sum()
    valid_indices2 = (~void_indices2).unsqueeze(1)
    num_valid2 = valid_indices2.sum()
    # channel_num*sum()/one_channel_valid already has a weight
    loss = (
        (-targets[:, :num_classes, :, :] * torch.log(MCLA + eps) -
         (1.0 - targets[:, :num_classes, :, :]) * torch.log(1.0 - MCMA + eps))
        * valid_indices).sum() / num_valid / num_classes
    loss += ((-targets_top[:, :, :, :] * torch.log(MCLB + eps) -
              (1.0 - targets_top[:, :, :, :]) * torch.log(1.0 - MCMB + eps)) *
             valid_indices2).sum() / num_valid2 / 7

    return 5 * loss


def losses_hiera_focal(predictions,
                       targets,
                       targets_top,
                       num_classes,
                       indices_high,
                       eps=1e-8,
                       gamma=2):
    """Implementation of hiera loss.

    Args:
        predictions (torch.Tensor): seg logits produced by decode head.
        targets (torch.Tensor): The learning label of the prediction.
        targets_top (torch.Tensor): The hierarchy ground truth of the learning
            label.
        num_classes (int): Number of categories.
        indices_high (List[List[int]]): Hierarchy indices of each hierarchy.
        eps (float):Term added to the Logarithm to improve numerical stability.
            Defaults: 1e-8.
        gamma (int): The exponent value. Defaults: 2.
    """
    b, _, h, w = predictions.shape
    predictions = torch.sigmoid(predictions.float())
    void_indices = (targets == 255)
    targets[void_indices] = 0
    targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)
    void_indices2 = (targets_top == 255)
    targets_top[void_indices2] = 0
    targets_top = F.one_hot(targets_top, num_classes=7).permute(0, 3, 1, 2)

    MCMA = predictions[:, :num_classes, :, :]
    MCMB = torch.zeros((b, 7, h, w),
                       dtype=predictions.dtype,
                       device=predictions.device)
    for ii in range(7):
        MCMB[:, ii:ii + 1, :, :] = torch.max(
            torch.cat([
                predictions[:, indices_high[ii][0]:indices_high[ii][1], :, :],
                predictions[:, num_classes + ii:num_classes + ii + 1, :, :]
            ],
                      dim=1), 1, True)[0]

    MCLB = predictions[:, num_classes:num_classes + 7, :, :]
    MCLA = predictions[:, :num_classes, :, :].clone()
    for ii in range(7):
        for jj in range(indices_high[ii][0], indices_high[ii][1]):
            MCLA[:, jj:jj + 1, :, :] = torch.min(
                torch.cat([
                    predictions[:, jj:jj + 1, :, :], MCLB[:, ii:ii + 1, :, :]
                ],
                          dim=1), 1, True)[0]

    valid_indices = (~void_indices).unsqueeze(1)
    num_valid = valid_indices.sum()
    valid_indices2 = (~void_indices2).unsqueeze(1)
    num_valid2 = valid_indices2.sum()
    # channel_num*sum()/one_channel_valid already has a weight
    loss = ((-targets[:, :num_classes, :, :] * torch.pow(
        (1.0 - MCLA), gamma) * torch.log(MCLA + eps) -
             (1.0 - targets[:, :num_classes, :, :]) * torch.pow(MCMA, gamma) *
             torch.log(1.0 - MCMA + eps)) *
            valid_indices).sum() / num_valid / num_classes
    loss += (
        (-targets_top[:, :, :, :] * torch.pow(
            (1.0 - MCLB), gamma) * torch.log(MCLB + eps) -
         (1.0 - targets_top[:, :, :, :]) * torch.pow(MCMB, gamma) *
         torch.log(1.0 - MCMB + eps)) * valid_indices2).sum() / num_valid2 / 7

    return 5 * loss


@LOSSES.register_module()
class HieraTripletLossCityscape(nn.Module):
    """Modified from https://github.com/qhanghu/HSSN_pytorch/blob/main/mmseg/mo
    dels/losses/hiera_triplet_loss_cityscape.py."""

    def __init__(self, num_classes, use_sigmoid=False, loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.treetripletloss = TreeTripletLoss(num_classes, hiera_map,
                                               hiera_index)
        self.ce = CrossEntropyLoss()

    def forward(self,
                step,
                embedding,
                cls_score_before,
                cls_score,
                label,
                weight=None,
                **kwargs):
        targets, targets_top, indices_top = prepare_targets(label)

        loss = losses_hiera(cls_score, targets, targets_top, self.num_classes,
                            indices_top)
        ce_loss = self.ce(cls_score[:, :-7], label)
        ce_loss2 = self.ce(cls_score[:, -7:], targets_top)
        loss = loss + ce_loss + ce_loss2

        loss_triplet, class_count = self.treetripletloss(embedding, label)
        class_counts = [
            torch.ones_like(class_count)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(class_counts, class_count, async_op=False)
        class_counts = torch.cat(class_counts, dim=0)

        if torch.distributed.get_world_size() == torch.nonzero(
                class_counts, as_tuple=False).size(0):
            factor = 1 / 4 * (1 + torch.cos(
                torch.tensor((step.item() - 80000) / 80000 *
                             math.pi))) if step.item() < 80000 else 0.5
            loss += factor * loss_triplet

        return loss * self.loss_weight
