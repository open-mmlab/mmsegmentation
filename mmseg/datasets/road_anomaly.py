# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp


import numpy as np
from mmcv.utils import print_log
from PIL import Image
from typing import Tuple
from .builder import DATASETS
from .custom import CustomDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from ..utils import get_ood_measures, print_measures_with_std, print_measures


@DATASETS.register_module()
class RoadAnomalyDataset(CustomDataset):
    """Road Anomaly dataset.
        THIS DATASETJUST FOR TESTING!!
        It has annotated the anomalies

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.labels/labels_semantic_converted.png',
                 **kwargs):
        super(RoadAnomalyDataset, self).__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        self.ood_indices = [25]

    def evaluate_ood(self, out_scores, in_scores) -> Tuple[np.float64, np.float64, np.float64]:
        if (len(out_scores) != 0) and (len(in_scores) != 0):
            auroc, apr, fpr = get_ood_measures(out_scores, in_scores)
            return auroc, apr, fpr
        else:
            print("This image does not contain any OOD pixels or is only OOD.")
            return None, None, None

    def get_in_out_conf(self, conf, index):
        seg_gt = self.get_gt_seg_map_by_idx(index)
        # Mask ignored index
        mask = seg_gt != self.ignore_index
        seg_gt = seg_gt[mask]
        conf = conf[mask]
        # Find out which pixels are OOD and which are not
        out_index = seg_gt == self.ood_indices[0]
        for label in self.ood_indices:
            out_index = np.logical_or(out_index, seg_gt == label)
        # gather their respective conf: neg because lower is better for detecting ood (alt. 1 - conf)
        in_scores = - conf[np.logical_not(out_index)]
        out_scores = - conf[out_index]
        return out_scores, in_scores

    def print_ood_measures(self, aurocs, auprs, fprs, logger=None):
        print_measures(aurocs, auprs, fprs, logger=logger)

    def print_ood_measures_with_std(self, aurocs, auprs, fprs, logger=None):
        print_measures_with_std(aurocs, auprs, fprs, logger=logger)

    def get_ood_mask(self, seg_gt):
        # Find out which pixels are OOD and which are not
        ood_mask = seg_gt == self.ood_indices[0]
        for label in self.ood_indices:
            ood_mask = np.logical_or(ood_mask, seg_gt == label)
        return (~ood_mask)
