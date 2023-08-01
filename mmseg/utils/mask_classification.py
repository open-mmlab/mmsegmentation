# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmcv.ops import point_sample
from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import TASK_UTILS
from mmseg.utils import ConfigType, SampleList


def seg_data_to_instance_data(ignore_index: int,
                              batch_data_samples: SampleList):
    """Perform forward propagation to convert paradigm from MMSegmentation to
    MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called normally.
    Specifically, ``batch_gt_instances`` would be added.

    Args:
        ignore_index (int): The label index to be ignored.
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
        batch_img_metas.append(data_sample.metainfo)
        gt_sem_seg = data_sample.gt_sem_seg.data
        classes = torch.unique(
            gt_sem_seg,
            sorted=False,
            return_inverse=False,
            return_counts=False)

        # remove ignored region
        gt_labels = classes[classes != ignore_index]

        masks = []
        for class_id in gt_labels:
            masks.append(gt_sem_seg == class_id)

        if len(masks) == 0:
            gt_masks = torch.zeros(
                (0, gt_sem_seg.shape[-2],
                 gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
        else:
            gt_masks = torch.stack(masks).squeeze(1).long()

        instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
        batch_gt_instances.append(instance_data)
    return batch_gt_instances, batch_img_metas


class MatchMasks:

    def __init__(self,
                 num_points: int,
                 num_queries: int,
                 num_classes: int,
                 assigner: ConfigType = None,
                 sampler: ConfigType = None):
        assert assigner is not None, "\'assigner\' in decode_head.train_cfg" \
                                     'cannot be None'
        assert num_points > 0, 'num_points should be a positive integer.'
        self.num_points = num_points
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.assigner = TASK_UTILS.build(assigner)
        self.sampler = TASK_UTILS.build(
            sampler, default_args=dict(context=self))

    def get_targets(self,
                    cls_scores: List[Tensor],
                    mask_preds: List[Tensor],
                    batch_gt_instances: List[InstanceData],
                    batch_img_metas: List[dict],
                    return_sampling_results: bool = False) -> Tuple:
        """Compute best mask matches for all images for a decoder layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - avg_factor (int): Average factor that is used to average\
                    the loss. When using sampling method, avg_factor is
                    usually the sum of positive and negative priors. When
                    using `MaskPseudoSampler`, `avg_factor` is usually equal
                    to the number of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end.
        """
        batch_size = cls_scores.shape[0]
        results = dict({
            'labels': [],
            'label_weights': [],
            'mask_targets': [],
            'mask_weights_list': [],
            'sampling_results': []
        })
        for i in range(batch_size):
            result = self._get_targets_single(cls_scores[i], mask_preds[i],
                                              batch_gt_instances[i],
                                              batch_img_metas[i])
            results['labels'].append(result[0])
            results['label_weights'].append(result[1])
            results['mask_targets'].append(result[2])
            results['mask_weights_list'].append(result[3])
            results['sampling_results'].append(result[6])

        labels = torch.stack(results['labels'], dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(results['label_weights'], dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(results['mask_targets'], dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(results['mask_weights_list'], dim=0)

        avg_factor = sum(
            [result.avg_factor for result in results['sampling_results']])

        res = (labels, label_weights, mask_targets, mask_weights, avg_factor)
        if return_sampling_results:
            res = res + (results['sampling_results'], )

        return res

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        """Compute a set of best mask matches for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)
