# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
"""Modified from https://github.com/PaddlePaddle/PaddleSeg/
blob/2c8c35a8949fef74599f5ec557d340a14415f20d/paddleseg/
models/losses/pixel_contrast_cross_entropy_loss.py(Apache-2.0 License)"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from mmseg.registry import MODELS


def hard_anchor_sampling(X: Tensor, y_hat: Tensor, y: Tensor,
                         ignore_index: int, max_views: int, max_samples: int):
    """
    Args:
        X (torch.Tensor): embedding, shape = [N, H * W, C]
        label (torch.Tensor): label, shape = [N, H * W]
        y_pred (torch.Tensor): predict mask, shape = [N, H * W]
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default 255.
        max_samples (int, optional): Max sampling anchors. Default: 1024.
        max_views (int): Sampled samplers of a class. Default: 100.
    Returns:
        tuple[Tensor]: A tuple contains two Tensors.
            - X_ (torch.Tensor): The sampled features,
                shape (total_classes, n_view, feat_dim).
            - y_ (torch.Tensor): The labels for X_ ,
                shape (total_classes, 1)
    """
    batch_size, feat_dim = X.shape[0], X.shape[-1]

    classes = []
    total_classes = 0
    for ii in range(batch_size):
        this_y = y_hat[ii]
        this_classes = torch.unique(this_y)
        this_classes = [x for x in this_classes if x != ignore_index]
        this_classes = [
            x for x in this_classes
            if (this_y == x).nonzero().shape[0] > max_views
        ]

        classes.append(this_classes)
        total_classes += len(this_classes)

    if total_classes == 0:
        return None, None

    n_view = max_samples // total_classes
    n_view = min(n_view, max_views)
    if (torch.cuda.is_available()):
        X_ = torch.zeros((total_classes, n_view, feat_dim),
                         dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
    else:
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float)
        y_ = torch.zeros(total_classes, dtype=torch.float)

    X_ptr = 0
    for ii in range(batch_size):
        this_y_hat = y_hat[ii]
        this_y = y[ii]
        this_classes = classes[ii]

        for cls_id in this_classes:
            hard_indices = ((this_y_hat == cls_id) &
                            (this_y != cls_id)).nonzero()
            easy_indices = ((this_y_hat == cls_id) &
                            (this_y == cls_id)).nonzero()

            num_hard = hard_indices.shape[0]
            num_easy = easy_indices.shape[0]

            if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                num_hard_keep = n_view // 2
                num_easy_keep = n_view - num_hard_keep
            elif num_hard >= n_view / 2:
                num_easy_keep = num_easy
                num_hard_keep = n_view - num_easy_keep
            elif num_easy >= n_view / 2:
                num_hard_keep = num_hard
                num_easy_keep = n_view - num_hard_keep
            else:
                num_hard_keep = num_hard
                num_easy_keep = num_easy

            perm = torch.randperm(num_hard)
            hard_indices = hard_indices[perm[:num_hard_keep]]
            perm = torch.randperm(num_easy)
            easy_indices = easy_indices[perm[:num_easy_keep]]
            indices = torch.cat((hard_indices, easy_indices), dim=0)

            X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
            y_[X_ptr] = cls_id
            X_ptr += 1

    return X_, y_


def contrastive(embed: Tensor, label: Tensor, temperature: float,
                base_temperature: float) -> Tensor:
    """
    Args:
        embed (torch.Tensor):
            sampled pixel, shape = [total_classes, n_view, feat_dim],
            total_classes = batch_size * single image classes
        label (torch.Tensor):
            The corresponding label for embed features, shape = [total_classes]
        temperature (float, optional):
            Controlling the numerical similarity of features.
            Default: 0.1.
        base_temperature (float, optional):
            Controlling the numerical range of contrast loss.
            Default: 0.07.

    Returns:
        loss (torch.Tensor): The calculated loss.
    """
    anchor_num, n_view = embed.shape[0], embed.shape[1]

    label = label.reshape((-1, 1))
    if (torch.cuda.is_available()):
        mask = torch.eq(label, label.permute([1, 0])).float().cuda()
    else:
        mask = torch.eq(label, label.permute([1, 0])).float()

    contrast_count = n_view
    contrast_feature = torch.cat(torch.unbind(embed, dim=1), dim=0)

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.permute([1, 0])),
        temperature)
    logits_max = torch.max(anchor_dot_contrast, dim=1, keepdim=True)[0]
    logits = anchor_dot_contrast - logits_max

    mask = torch.tile(mask, [anchor_count, contrast_count])
    neg_mask = 1 - mask

    if (torch.cuda.is_available()):
        logits_mask = torch.ones_like(mask).scatter_(
            1,
            torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(), 0)
    else:
        logits_mask = torch.ones_like(mask).scatter_(
            1,
            torch.arange(anchor_num * anchor_count).view(-1, 1), 0)

    mask = mask * logits_mask

    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)

    exp_logits = torch.exp(logits)

    log_prob = logits - torch.log(exp_logits + neg_logits)

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    loss = -(temperature / base_temperature) * mean_log_prob_pos
    loss = loss.mean()

    return loss


def contrast_criterion(
    feats: Tensor,
    labels: Tensor,
    predict: Tensor,
    ignore_index=255,
    max_views=100,
    max_samples=1024,
    temperature=0.1,
    base_temperature=0.07,
) -> Tensor:
    '''
    Args:
        feats (torch.Tensor): embedding, shape = [N, H * W, C]
        labels (torch.Tensor): label, shape = [N, H * W]
        predict (torch.Tensor): predict mask, shape = [N, H * W]
        ignore_index (int, optional):
            Specifies a target value that is ignored
            and does not contribute to the input gradient.
            Default 255.
        max_samples (int, optional): Max sampling anchors.
        Default: 1024.
        max_views (int): Sampled samplers of a class. Default: 100.
        temperature (float):A hyper-parameter in contrastive loss,
            controlling the numerical similarity of features.
            Default: 0.1.
        base_temperature (float):A hyper-parameter in contrastive loss,
            controlling the numerical range of contrast loss.
            Default: 0.07.
    Returns:
        loss (torch.Tensor): The calculated loss
    '''
    labels = labels.unsqueeze(1).float().clone()
    labels = torch.nn.functional.interpolate(
        labels, (feats.shape[2], feats.shape[3]), mode='nearest')
    labels = labels.squeeze(1).long()
    assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(
        labels.shape, feats.shape)

    batch_size = feats.shape[0]
    labels = labels.reshape((batch_size, -1))
    predict = predict.reshape((batch_size, -1))
    feats = feats.permute([0, 2, 3, 1])
    feats = feats.reshape((feats.shape[0], -1, feats.shape[-1]))

    feats_, labels_ = hard_anchor_sampling(feats, labels, predict,
                                           ignore_index, max_views,
                                           max_samples)

    loss = contrastive(
        feats_,
        labels_,
        temperature,
        base_temperature,
    )
    return loss


@MODELS.register_module()
class PixelContrastCrossEntropyLoss(nn.Module):
    """The PixelContrastCrossEntropyLoss is proposed in "Exploring Cross-Image
    Pixel Contrast for Semantic Segmentation"
    (https://arxiv.org/abs/2101.11939) Wenguan Wang, Tianfei Zhou, et al..

    Args:
        loss_name (str, optional):
            Name of the loss item.
            If you want this loss item to be included into the backward graph,
            `loss_` must be the prefix of the name.
            Defaults to 'loss_pixel_contrast_cross_entropy'.
        temperature (float, optional):
            Controlling the numerical similarity of features.
            Default: 0.1.
        base_temperature (float, optional):
            Controlling the numerical range of contrast loss.
            Default: 0.07.
        ignore_index (int, optional):
            Specifies a target value that is ignored
            and does not contribute to the input gradient.
            Default 255.
        max_samples (int, optional):
            Max sampling anchors. Default: 1024.
        max_views (int):
            Sampled samplers of a class. Default: 100.
    """

    def __init__(self,
                 loss_name='loss_pixel_contrast_cross_entropy',
                 temperature=0.1,
                 base_temperature=0.07,
                 ignore_index=255,
                 max_samples=1024,
                 max_views=100,
                 loss_weight=0.1):
        super().__init__()
        self._loss_name = loss_name
        self.loss_weight = loss_weight
        if (temperature < 0 or base_temperature <= 0):
            raise KeyError(
                'temperature should >=0 and base_temperature should >0')
        self.temperature = temperature
        self.base_temperature = base_temperature
        if (not isinstance(ignore_index, int) or ignore_index < 0
                or ignore_index > 255):
            raise KeyError('ignore_index should be an int between 0 and 255')
        self.ignore_index = ignore_index
        if (max_samples <= 0 or not isinstance(max_samples, int)):
            raise KeyError('max_samples should be an int and >=0')
        self.max_samples = max_samples
        if (max_views <= 0 or not isinstance(max_views, int)):
            raise KeyError('max_views should be an int and >=0')
        self.max_views = max_views

    def forward(self, pred: List, target: Tensor) -> Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape
                (N, C) where C = number of classes, or
                (N, C, d_1, d_2, ..., d_K) with K≥1 in the
                case of K-dimensional loss.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1,
                or (N, d_1, d_2, ..., d_K) with K≥1 in the case of
                K-dimensional loss. If containing class probabilities,
                same shape as the input.

        Returns:
            torch.Tensor: The calculated loss
        """

        assert isinstance(pred, list) and len(pred) == 2, 'Only ContrastHead \
                is suitable for PixelContrastCrossEntropyLoss'

        seg = pred[0]
        embedding = pred[1]

        predict = torch.argmax(seg, dim=1)

        loss = contrast_criterion(embedding, target, predict,
                                  self.ignore_index, self.max_views,
                                  self.max_samples, self.temperature,
                                  self.base_temperature)

        return loss * self.loss_weight

    @property
    def loss_name(self) -> str:
        """Loss Name. This function must be implemented and will return the
        name of this loss function. This name will be used to combine different
        loss items by simple sum operation. In addition, if you want this loss
        item to be included into the backward graph, `loss_` must be the prefix
        of the name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
