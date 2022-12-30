# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

try:
    from mmdet.models.dense_heads import MaskFormerHead as MMDET_MaskFormerHead
except ModuleNotFoundError:
    MMDET_MaskFormerHead = BaseModule

from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class MaskFormerHead(MMDET_MaskFormerHead):
    """Implements the MaskFormer head.

    See `Per-Pixel Classification is Not All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`_ for details.

    Args:
        num_classes (int): Number of classes. Default: 150.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 num_classes: int = 150,
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.out_channels = kwargs['out_channels']
        self.align_corners = True
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index

        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_MaskFormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []
        for data_sample in batch_data_samples:
            # Add `batch_input_shape` in metainfo of data_sample, which would
            # be used in MaskFormerHead of MMDetection.
            metainfo = data_sample.metainfo
            metainfo['batch_input_shape'] = metainfo['img_shape']
            data_sample.set_metainfo(metainfo)
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros((0, gt_sem_seg.shape[-2],
                                        gt_sem_seg.shape[-1])).to(gt_sem_seg)
            else:
                gt_masks = torch.stack(masks).squeeze(1)

            instance_data = InstanceData(
                labels=gt_labels, masks=gt_masks.long())
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """

        batch_data_samples = []
        for metainfo in batch_img_metas:
            metainfo['batch_input_shape'] = metainfo['img_shape']
            batch_data_samples.append(SegDataSample(metainfo=metainfo))
        # Forward function of MaskFormerHead from MMDetection needs
        # 'batch_data_samples' as inputs, which is image shape　actually.
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=img_shape,
            mode='bilinear',
            align_corners=False)

        # semantic inference
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
        return seg_logits
