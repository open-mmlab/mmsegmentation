import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import get_class_count, weight_reduce_loss
import numpy as np
import torch.nn.functional as F


@LOSSES.register_module
class LDAMLoss(nn.Module):
    """
    Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    Works only with ignore index
    """

    def __init__(
            self, class_count, max_margin=0.5, class_weight=None, scale=30, reduction="mean", warm=50, loss_weight=1.0, avg_non_ignore=False,
            use_bags=True, loss_name='loss_ce_ldam'):

        super(LDAMLoss, self).__init__()
        if self.warm > 0:
            self.loss_name = "loss_ce"
        else:
            self.loss_name = "loss_ldam"
        self.loss_name = loss_name
        self.scale = scale
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        self.warm = warm
        self.epoch_num = 0
        self.use_bags = False
        if not use_bags:
            self.class_count = get_class_count(class_count)[:-1]  # last item is for 255 (bg)
            delta = 1.0 / (self.class_count**0.25)
            delta = max_margin * (delta / np.max(delta))
            self.delta = torch.cuda.FloatTensor(delta)
        else:
            # compute deltas for every bag
            self.class_count = class_count
            assert isinstance(class_count, list) and isinstance(class_count[0], np.ndarray)
            self.delta = []
            for bg_cls_cnt in class_count:
                delta_ = 1.0 / (bg_cls_cnt**0.25)
                delta_ = max_margin * (delta_ / np.max(delta_))
                delta_ = torch.cuda.FloatTensor(delta_)
                self.delta.append(delta_)

    def forward(self,
                pred,  # logits
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                bag_idx=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (reduction_override if reduction_override else self.reduction)

        if self.epoch_num > self.warm:
            self.loss_name = 'loss_ldam'
            if self.use_bags:
                assert bag_idx is not None and bag_idx < len(self.delta)
                delta = self.delta[bag_idx]
            else:
                delta = self.delta

            batch_target_ = target.data.unsqueeze(1).clone()
            # Label 255 is causing this error in gather and scatter_
            batch_target_[batch_target_ == 255] = 0

            target_expanded = target.data.unsqueeze(1).clone()
            mask_ignore = (target_expanded == 255)
            batch_target_[mask_ignore] = 0

            delta_ = delta[None, :].reshape(1, -1, 1, 1).repeat(batch_target_.shape)
            batch_delta = torch.gather(input=delta_, dim=1, index=batch_target_)
            diff = pred - batch_delta
            one_hot_gt = torch.zeros_like(pred, dtype=torch.uint8).scatter_(1, batch_target_, 1)
            output = torch.where(one_hot_gt.type(torch.bool), diff, pred)
            loss = F.cross_entropy(self.scale * output, target, weight=self.class_weight, reduction="none", ignore_index=ignore_index)
        else:
            loss = F.cross_entropy(pred, target, weight=self.class_weight, reduction="none", ignore_index=ignore_index)

        avg_factor = target.numel() - (target == ignore_index).sum().item()
        if reduction == 'mean':
            loss_cls = loss.sum() / avg_factor
        elif reduction == 'sum':
            loss_cls = loss.sum()
        else:
            loss_cls = loss

        return self.loss_weight * loss_cls
