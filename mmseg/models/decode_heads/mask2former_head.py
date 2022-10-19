# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import Mask2FormerHead as Mask2FormerHead_
from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class Mask2FormerHead(Mask2FormerHead_):

    def __init__(self, num_classes, align_corners=False, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        # self.num_things_classes = num_classes
        # self.num_stuff_classes = 0
        self.align_corners = align_corners
        self.out_channels = num_classes

        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, self.num_classes)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        batch_img_metas = []
        batch_gt_instances = []
        # batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_semantic_seg = data_sample.gt_sem_seg.data
            gt_labels = torch.unique(
                gt_semantic_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)
            gt_masks = torch.stack(
                [gt_semantic_seg == label for label in gt_labels]).squeeze(1)

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
            # batch_gt_semantic_segs.append(data_sample.gt_sem_seg)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor], batch_data_samples: SampleList,
                test_cfg: ConfigType) -> Tuple[Tensor]:

        # batch_img_metas = [
        #     data_sample.metainfo for data_sample in batch_data_samples
        # ]
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        sem_seg_mask = torch.einsum('bqc, qhw->bchw', cls_score, mask_pred)
        return sem_seg_mask
