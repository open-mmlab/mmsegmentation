# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import calc_sod_metrics, eval_sod_metrics, \
    pre_eval_to_sod_metrics
from . import CustomDataset
from .builder import DATASETS


@DATASETS.register_module()
class SODCustomDataset(CustomDataset):
    CLASSES = None

    PALETTE = None

    def __init__(self, **kwargs):
        super(SODCustomDataset, self).__init__(**kwargs)

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pre_eval_results.append(
                calc_sod_metrics(pred, seg_map))

        return pre_eval_results

    def evaluate(self,
                 results,
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            ret_metrics = eval_sod_metrics(
                results,
                gt_seg_maps)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_sod_metrics(results)

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            summary_table_data.add_column(key, [val])

        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            eval_results[key] = value / 100.0

        return eval_results
