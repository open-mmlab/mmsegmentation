# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp


import numpy as np
from mmcv.utils import print_log
from PIL import Image
from typing import Tuple
from .builder import DATASETS
from .custom import CustomDataset
from ..utils import get_ood_measures, print_measures_with_std, print_measures


@DATASETS.register_module()
class StreetHazardsDataset(CustomDataset):
    """
    Street Hazards dataset.
    """

    CLASSES = (
        # "unlabeled",
        'building',
        'fence',
        'other',
        'pedestrian',
        'pole',
        'road line',
        'road',
        'sidewalk',
        'vegetation',
        'car',
        'wall',
        'trafic sign',
        # "anomaly"
    )

    PALETTE = [
        # [0, 0, 0],  # unlabeled
        [70, 70, 70],
        [190, 153, 153],
        [250, 170, 160],
        [220, 20, 60],
        [153, 153, 153],
        [157, 234, 50],
        [128, 64, 128],
        [244, 35, 232],
        [107, 142, 35],
        [0, 0, 142],
        [102, 102, 156],
        [220, 220, 0],
        # [60, 250, 240]  # Anomaly
    ]

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs):
        super(StreetHazardsDataset, self).__init__(img_suffix=img_suffix,
                                                   seg_map_suffix=seg_map_suffix,
                                                   reduce_zero_label=True,
                                                   #    gt_seg_map_loader_cfg={"reduce_zero_label": True},
                                                   **kwargs)
        self.custom_classes = True
        self.label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13}
        self.ood_indices = [12]
        self.num_classes = 12

    def evaluate_ood(self, out_scores, in_scores) -> Tuple[np.float64, np.float64, np.float64]:
        if (len(out_scores) != 0) and (len(in_scores) != 0):
            auroc, apr, fpr, ece = get_ood_measures(out_scores, in_scores)
            return auroc, apr, fpr, ece
        else:
            return None, None, None, None

    def get_in_out_conf(self, conf, index):
        in_scores = {}
        out_scores = {}
        seg_gt = self.get_gt_seg_map_by_idx(index)
        if self.reduce_zero_label:
            seg_gt[seg_gt == 0] = 255
            seg_gt = seg_gt - 1
            seg_gt[seg_gt == 254] = 255
        # Mask ignored index
        mask = (seg_gt != self.ignore_index)
        seg_gt = seg_gt[mask]
        # Find out which pixels are OOD and which are not
        out_index = (seg_gt == self.ood_indices[0])
        for label in self.ood_indices:
            out_index = np.logical_or(out_index, (seg_gt == label))
        for k in conf.keys():
            conf[k] = conf[k].squeeze()[mask]
            if k in ("max_softmax", "max_logit"):
                # gather their respective conf values
                in_scores[k] = - conf[k][np.logical_not(out_index)]
                out_scores[k] = - conf[k][out_index]
            elif k == "entropy":
                in_scores[k] = conf[k][np.logical_not(out_index)]
                out_scores[k] = conf[k][out_index]
            else:
                raise KeyError(k)

        return out_scores, in_scores

    def print_ood_measures(self, aurocs, auprs, fprs, eces, logger=None, text="max_softmax"):
        print_measures(aurocs, auprs, fprs, eces, logger=logger, text=text)

    def print_ood_measures_with_std(self, aurocs, auprs, fprs, eces, logger=None, text="max_softmax"):
        print_measures_with_std(aurocs, auprs, fprs, eces, logger=logger, text=text)

    def get_ood_masker(self, seg_gt):
        # Find out which pixels are OOD and which are not
        if self.reduce_zero_label:
            seg_gt[seg_gt == 0] = 255
            seg_gt = seg_gt - 1
            seg_gt[seg_gt == 254] = 255
        ood_mask = seg_gt == self.ood_indices[0]
        for label in self.ood_indices:
            ood_mask = np.logical_or(ood_mask, seg_gt == label)
        return (~ood_mask)

    def get_class_count(self, path="."):
        class_count_pixel = {i: 0 for i in range(len(self.CLASSES))}
        class_count_pixel[255] = 0  # ignore background
        class_count_semantic = {i: 0 for i in range(len(self.CLASSES))}
        class_count_semantic[255] = 0  # ignore background
        for index in range(self.__len__()):
            seg_gt = self.get_gt_seg_map_by_idx(index)
            if self.reduce_zero_label:
                seg_gt[seg_gt == 0] = 255
                seg_gt = seg_gt - 1
                seg_gt[seg_gt == 254] = 255
            for i in class_count_pixel.keys():
                class_count_pixel[i] += int((seg_gt == i).sum())
                class_count_semantic[i] += int((np.unique(seg_gt) == i).sum())
        class_count_pixel = np.array([*class_count_pixel.values()])
        class_count_semantic = np.array([*class_count_semantic.values()])

        filename = f"class_count_street_hazards_pixel.npy"
        with open(osp.join(path, filename), "wb") as f:
            np.save(f, class_count_pixel)

        filename = f"class_count_street_hazards_semantic.npy"
        with open(osp.join(path, filename), "wb") as f:
            np.save(f, class_count_semantic)
