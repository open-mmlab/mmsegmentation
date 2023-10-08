# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from prettytable import PrettyTable
from torch import Tensor

from mmseg.registry import METRICS


@METRICS.register_module()
class DepthMetric(BaseMetric):
    """Depth estimation evaluation metric.

    Args:
        depth_metrics (List[str], optional): List of metrics to compute. If
            not specified, defaults to all metrics in self.METRICS.
        min_depth_eval (float): Minimum depth value for evaluation.
            Defaults to 0.0.
        max_depth_eval (float): Maximum depth value for evaluation.
            Defaults to infinity.
        crop_type (str, optional): Specifies the type of cropping to be used
            during evaluation. This option can affect how the evaluation mask
            is generated. Currently, 'nyu_crop' is supported, but other
            types can be added in future. Defaults to None if no cropping
            should be applied.
        depth_scale_factor (float): Factor to scale the depth values.
            Defaults to 1.0.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    METRICS = ('d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog')

    def __init__(self,
                 depth_metrics: Optional[List[str]] = None,
                 min_depth_eval: float = 0.0,
                 max_depth_eval: float = float('inf'),
                 crop_type: Optional[str] = None,
                 depth_scale_factor: float = 1.0,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if depth_metrics is None:
            self.metrics = self.METRICS
        elif isinstance(depth_metrics, [tuple, list]):
            for metric in depth_metrics:
                assert metric in self.METRICS, f'the metric {metric} is not ' \
                    f'supported. Please use metrics in {self.METRICS}'
            self.metrics = depth_metrics

        # Validate crop_type, if provided
        assert crop_type in [
            None, 'nyu_crop'
        ], (f'Invalid value for crop_type: {crop_type}. Supported values are '
            'None or \'nyu_crop\'.')
        self.crop_type = crop_type
        self.min_depth_eval = min_depth_eval
        self.max_depth_eval = max_depth_eval
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.depth_scale_factor = depth_scale_factor

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_depth_map']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                gt_depth = data_sample['gt_depth_map']['data'].squeeze().to(
                    pred_label)

                eval_mask = self._get_eval_mask(gt_depth)
                self.results.append(
                    (gt_depth[eval_mask], pred_label[eval_mask]))
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy(
                ) * self.depth_scale_factor

                cv2.imwrite(png_filename, output_mask.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def _get_eval_mask(self, gt_depth: Tensor):
        """Generates an evaluation mask based on ground truth depth and
        cropping.

        Args:
            gt_depth (Tensor): Ground truth depth map.

        Returns:
            Tensor: Boolean mask where evaluation should be performed.
        """
        valid_mask = torch.logical_and(gt_depth > self.min_depth_eval,
                                       gt_depth < self.max_depth_eval)

        if self.crop_type == 'nyu_crop':
            # this implementation is adapted from
            # https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/depth/datasets/nyu.py  # noqa
            crop_mask = torch.zeros_like(valid_mask)
            crop_mask[45:471, 41:601] = 1
        else:
            crop_mask = torch.ones_like(valid_mask)

        eval_mask = torch.logical_and(valid_mask, crop_mask)
        return eval_mask

    @staticmethod
    def _calc_all_metrics(gt_depth, pred_depth):
        """Computes final evaluation metrics based on accumulated results."""
        assert gt_depth.shape == pred_depth.shape

        thresh = torch.max((gt_depth / pred_depth), (pred_depth / gt_depth))
        diff = pred_depth - gt_depth
        diff_log = torch.log(pred_depth) - torch.log(gt_depth)

        d1 = torch.sum(thresh < 1.25).float() / len(thresh)
        d2 = torch.sum(thresh < 1.25**2).float() / len(thresh)
        d3 = torch.sum(thresh < 1.25**3).float() / len(thresh)

        abs_rel = torch.mean(torch.abs(diff) / gt_depth)
        sq_rel = torch.mean(torch.pow(diff, 2) / gt_depth)

        rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
        rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

        log10 = torch.mean(
            torch.abs(torch.log10(pred_depth) - torch.log10(gt_depth)))
        silog = torch.sqrt(
            torch.pow(diff_log, 2).mean() -
            0.5 * torch.pow(diff_log.mean(), 2))

        return {
            'd1': d1.item(),
            'd2': d2.item(),
            'd3': d3.item(),
            'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(),
            'rmse': rmse.item(),
            'rmse_log': rmse_log.item(),
            'log10': log10.item(),
            'silog': silog.item()
        }

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The keys
                are identical with self.metrics.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        metrics = defaultdict(list)
        for gt_depth, pred_depth in results:
            for key, value in self._calc_all_metrics(gt_depth,
                                                     pred_depth).items():
                metrics[key].append(value)
        metrics = {k: sum(metrics[k]) / len(metrics[k]) for k in self.metrics}

        table_data = PrettyTable()
        for key, val in metrics.items():
            table_data.add_column(key, [round(val, 5)])

        print_log('results:', logger)
        print_log('\n' + table_data.get_string(), logger=logger)

        return metrics
