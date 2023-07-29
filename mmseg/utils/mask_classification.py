# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.structures import InstanceData

from mmseg.utils import SampleList


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
