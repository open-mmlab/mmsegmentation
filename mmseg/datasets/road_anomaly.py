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
import torch
from copy import deepcopy


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
        super(RoadAnomalyDataset, self).__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,
                                                 reduce_zero_label=False, **kwargs)
        self.ood_indices = [25]  # road anomaly has two labels 0 and 25
        self.num_classes = 19  # cityscapes

    def evaluate_ood(self, out_scores, in_scores) -> Tuple[np.float64, np.float64, np.float64]:
        if (len(out_scores) != 0) and (len(in_scores) != 0):
            auroc, apr, fpr, ece = get_ood_measures(out_scores, in_scores)
            return auroc, apr, fpr, ece
        else:
            return None, None, None, None

    def get_in_out_conf(self, pred_confs, seg_gt, conf_type):
        assert conf_type in ("max_prob", "max_logit", "entropy")
        confs = deepcopy(pred_confs)
        # Mask ignored index
        mask = (seg_gt != self.ignore_index)
        seg_gt = seg_gt[mask]
        # Find out which pixels are OOD and which are not
        out_index = (seg_gt == self.ood_indices[0])
        for label in self.ood_indices:
            out_index = np.logical_or(out_index, (seg_gt == label))
        confs = confs.squeeze()[mask]
        if conf_type in ("max_prob", "max_logit"):
            # gather their respective conf values
            in_scores = - confs[np.logical_not(out_index)]
            out_scores = - confs[out_index]
        else:
            in_scores = confs[np.logical_not(out_index)]
            out_scores = confs[out_index]
        return out_scores, in_scores

    def print_ood_measures(self, aurocs, auprs, fprs, eces, logger=None, text="max_softmax"):
        print_measures(aurocs, auprs, fprs, eces, logger=logger, text=text)

    def print_ood_measures_with_std(self, aurocs, auprs, fprs, eces, logger=None, text="max_softmax"):
        print_measures_with_std(aurocs, auprs, fprs, eces, logger=logger, text=text)

    def get_ood_masker(self, seg_gt):
        # Find out which pixels are OOD and which are not
        ood_mask = seg_gt == self.ood_indices[0]
        for label in self.ood_indices:
            ood_mask = np.logical_or(ood_mask, seg_gt == label)
        return (~ood_mask)

    def edge_detector(self, seg_gt, kernel_size=2):
        seg_gt = torch.from_numpy(seg_gt)
        assert len(seg_gt.size()) == 2
        stride = kernel_size

        patches = seg_gt.unfold(0, kernel_size, stride).unfold(1, kernel_size, stride)
        unfold_shape = patches.size()

        # DO Whatever ops that doesn't change the shape
        ones = torch.ones_like(patches, dtype=torch.bool)
        zeros = torch.zeros_like(patches, dtype=torch.bool)
        patches_eq_elems = (patches == patches[:, :, 0:1, 0:1]).all(-1, True).all(-2, True).repeat(1, 1, kernel_size, kernel_size)
        patches = torch.where(patches_eq_elems, zeros, ones)

        assert patches.size() == unfold_shape

        patches = patches.contiguous().view(-1, kernel_size, kernel_size)

        # Reshape back
        patches_orig = patches.view(unfold_shape)
        output_h = unfold_shape[0] * unfold_shape[2]
        output_w = unfold_shape[1] * unfold_shape[3]
        patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()
        patches_orig = patches_orig.view(output_h, output_w)

        assert patches_orig.size() == seg_gt.size()
        return patches_orig

    def unfold_fold(self, seg_gt, kernel_size=2, stride=2):
        seg_gt = torch.from_numpy(seg_gt)
        assert len(seg_gt.size()) == 2

        patches = seg_gt.unfold(0, kernel_size, stride).unfold(1, kernel_size, stride)
        unfold_shape = patches.size()

        # DO Whatever ops that doesn't change the shape

        patches = patches.contiguous().view(-1, kernel_size, kernel_size)

        # Reshape back
        patches_orig = patches.view(unfold_shape)
        output_h = unfold_shape[0] * unfold_shape[2]
        output_w = unfold_shape[1] * unfold_shape[3]
        patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()
        patches_orig = patches_orig.view(output_h, output_w)

        assert (patches_orig == seg_gt).all()

    def get_bags(self, mul=10):
        if not hasattr(self, "class_count_pixel"):
            try:
                with open("class_count_cityscapes_pixel.npy", "rb") as f:
                    self.class_count_pixel = np.load(f)
            except FileNotFoundError as e:
                raise e

        class_count = self.class_count_pixel[:-1]
        low = float(1 / mul)
        hi = float(mul)
        used = np.zeros(self.num_classes, dtype=bool)
        bag_masks = []
        ratios = []
        bag_index = 0
        label2bag = {}
        bag_label_maps = []
        bags_classes = []
        bag_class_counts = []
        for cls in range(self.num_classes):
            if used[cls]:
                continue
            ratio_ = class_count / class_count[cls]
            ratios.append(ratio_)
            bag_mask = np.logical_and((ratio_ >= low), (ratio_ <= hi))
            if np.logical_and(bag_mask, used).any():
                # check if conflicts with used
                for c in np.where(np.logical_and(bag_mask, used))[0]:
                    conflict_bag_idx = label2bag[c]
                    conflict_bag_mask = bag_masks[conflict_bag_idx]
                    if bag_mask.sum() > conflict_bag_mask.sum():
                        bag_mask[c] = False
                    else:
                        conflict_bag_mask[c] = False
                        bag_masks[conflict_bag_idx] = conflict_bag_mask
            used = np.logical_or(used, bag_mask)
            bag_masks.append(bag_mask)

            for c in np.where(bag_mask)[0]:
                label2bag[c] = bag_index

            bag_index += 1
        num_bags = len(bag_masks)

        for i in range(num_bags):
            label_map = []
            for c in range(self.num_classes):
                if bag_masks[i][c]:
                    label_map.append(c)
                else:
                    label_map.append(self.num_classes + i)
            bag_label_maps.append(label_map)

            bag_clas_count = class_count[bag_masks[i]]
            bag_clas_count = np.append(bag_clas_count, class_count[~bag_masks[i]].sum())
            bag_class_counts.append(bag_clas_count)

            oth_mask = np.zeros(num_bags, dtype=bool)
            oth_mask[i] = True
            bag_masks[i] = np.concatenate((bag_masks[i], oth_mask))
            bags_classes.append([*np.where(bag_masks[i])[0]])

        assert all([bag_class_count.sum() == class_count.sum() for bag_class_count in bag_class_counts])
        assert np.sum([bag_mask.sum() for bag_mask in bag_masks]) == self.num_classes + num_bags

        self.num_bags = num_bags
        self.label2bag = label2bag
        self.bag_label_maps = bag_label_maps
        self.bag_masks = bag_masks
        self.bag_class_counts = bag_class_counts
        self.bags_classes = bags_classes
