# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS


@METRICS.register_module()
class IoUMetricG(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
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

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                self.results.append(
                    (pred_label, label)
                )
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 2

        # total_area_intersect = sum(results[0])
        # total_area_union = sum(results[1])
        # total_area_pred_label = sum(results[2])
        # total_area_label = sum(results[3])
        ret_metrics = self.intersect_and_union(
            results[0], results[1])

        print(ret_metrics)

        # # summary table
        # ret_metrics_summary = OrderedDict({
        #     ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        #     for ret_metric, ret_metric_value in ret_metrics.items()
        # })
        # metrics = dict()
        # for key, val in ret_metrics_summary.items():
        #     if key == 'aAcc':
        #         metrics[key] = val
        #     else:
        #         metrics['m' + key] = val

        # # each class table
        # ret_metrics.pop('aAcc', None)
        # ret_metrics_class = OrderedDict({
        #     ret_metric: np.round(ret_metric_value * 100, 2)
        #     for ret_metric, ret_metric_value in ret_metrics.items()
        # })
        # ret_metrics_class.update({'Class': class_names})
        # ret_metrics_class.move_to_end('Class', last=False)
        # class_table_data = PrettyTable()
        # for key, val in ret_metrics_class.items():
        #     class_table_data.add_column(key, val)

        # print_log('per class results:', logger)
        # print_log('\n' + class_table_data.get_string(), logger=logger)
        # return ret_metrics
        print_log(ret_metrics)
        return ret_metrics

    @staticmethod
    def intersect_and_union(pred_labels: torch.tensor, labels: torch.tensor):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        
        iou_list = []
        for pred, mask in zip(pred_labels, labels):
            print(f"{pred_labels.dtype}, {labels.dtype}")
            # pred: (H, W): bool, mask: (H, W): bool
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
        iou_list = np.stack(iou_list)
        iou_list = torch.from_numpy(iou_list).cpu()
        prec_list = []
        for thres in torch.arange(0.5, 1.0, 0.1):
            tmp = (iou_list > thres).float().mean()
            prec_list.append(tmp)
        iou = iou_list.mean()
        prec = {}
        temp = '  '
        for i, thres in enumerate(range(5, 10)):
            key = 'Pr@{}'.format(thres * 10)
            value = prec_list[i].item()
            prec[key] = value
            temp += "{}: {:.2f}  ".format(key, 100. * value)
        head = 'Evaluation: IoU={:.2f}'.format(100. * iou.item())
        print("ret")
        print({'iou': iou.item(), **prec})
        return head + temp, {'iou': iou.item(), **prec}

   