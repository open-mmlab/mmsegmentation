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
from my_projects.conversion_tests.converters.dataset_converter import(
    DatasetConverter
)


@METRICS.register_module()
class CustomIoUMetricConversion(BaseMetric):
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
    # TODO let converter determine reduce zero_label
    def __init__(self,
        test_dataset: str = "HOTS",
        output_dataset: str = "ADE20K",
        target_dataset: str = "HOTS_CAT",
        ignore_index: int = 255,
        iou_metrics: List[str] = ['mIoU'],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
        collect_device: str = 'cpu',
        output_dir: Optional[str] = None,
        format_only: bool = False,
        prefix: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.data_converter = DatasetConverter(
            test_dataset=test_dataset,
            output_dataset=output_dataset,
            target_dataset=target_dataset
        )
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
        # print(f"databatch: {data_batch}")
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            # if not self.format_only:
            label = data_sample['gt_sem_seg']['data'].squeeze().to(
                pred_label)
            pred_label, label = self.data_converter.convert_labels(
                pred_label=pred_label,
                gt_label=label
            )
            pred_label = torch.tensor(pred_label)
            label = torch.tensor(label)
            self.results.append(
                self.intersect_and_union(
                    pred_label=pred_label,
                    label=label,
                    ignore_index=self.ignore_index,
                    dataset_converter=self.data_converter
                )
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
        results = tuple(zip(*results))
        assert len(results) == 4
       
        area_intersect = np.asarray([np.array(res) for res in results[0]])
        area_union = np.asarray([np.array(res) for res in results[1]])
        area_pred_label = np.asarray([np.array(res) for res in results[2]])
        area_label = np.asarray([np.array(res) for res in results[3]])
        
        ret_metrics = self.total_area_to_metrics(
            area_intersect, area_union, area_pred_label,
            area_label, self.metrics, self.nan_to_num, self.beta)

        # ret_metrics["IoU_mm"] = ret_iou_mm["IoU_mm"]
        
        class_names = DatasetConverter.get_class_names(
            dataset_name=self.data_converter.target_dataset
        )
        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def intersect_and_union(
        pred_label: torch.tensor, 
        label: torch.tensor,
        ignore_index: int,
        dataset_converter: DatasetConverter
    ):
        """Calculate Intersection and Union.

        

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        num_classes = len(
            dataset_converter.get_class_names(
                dataset_name=dataset_converter.target_dataset
            )
        )
        # TODO check
        # pred_label, label = dataset_converter.convert_labels(
        #     pred_label=pred_label,
        #     gt_label=label
        # )
        # pred_label = torch.tensor(pred_label)
        # label = torch.tensor(label)
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label


    @staticmethod
    def total_area_to_metrics(area_intersect: np.ndarray,
                              area_union: np.ndarray,
                              area_pred_label: np.ndarray,
                              area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes. 
            area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """
        
        n_samples = area_intersect.shape[0]
        n_classes = area_intersect.shape[1]
        
        ret_metrics = OrderedDict()
        # iou per sample
        for thres in np.arange(0.5, 1.0, 0.1):
            class_scores = np.zeros((n_classes))
            class_counts = np.zeros((n_classes))
            for sample_idx in range(n_samples):
                
                inter = area_intersect[sample_idx]
                union = area_union[sample_idx]
                pred = area_pred_label[sample_idx]
                gt = area_label[sample_idx]
                iou = inter / union
                class_scores += (iou > thres).astype(np.int32)
                class_counts += (gt > 0).astype(np.int32)
            class_scores /= class_counts
            pr_key = f"Pr@{np.round(thres * 100, 2).astype(np.int32)}"
            ret_metrics[pr_key] = class_scores
                
                
                

        
           
        total_iou = np.sum(area_intersect, 0) / np.sum(area_union, 0) 
        ret_metrics["IoU"] = total_iou
        
        
        
        # ret_metrics = {
        #     metric: value.numpy()
        #     for metric, value in ret_metrics.items()
        # }
        # if nan_to_num is not None:
        # ret_metrics = OrderedDict({
        #     metric: np.nan_to_num(metric_value, nan=0.0)
        #     for metric, metric_value in ret_metrics.items()
        # })
        # print("ret: ")
        # for key, val in ret_metrics.items():
        #     print(f"{key} : {val}")
        # print()
        
        return ret_metrics

    @staticmethod
    def mm_iou(
        total_area_intersect: np.ndarray,
        total_area_union: np.ndarray
    ):
        iou = total_area_intersect / total_area_union
        ret_metrics = OrderedDict({'IoU_mm': iou})
        
        
        
        ret_metrics = {
            metric: np.asarray(value)
            for metric, value in ret_metrics.items()
        }
        # if nan_to_num is not None:
        #     ret_metrics = OrderedDict({
        #         metric: np.nan_to_num(metric_value, nan=nan_to_num)
        #         for metric, metric_value in ret_metrics.items()
        #     })
        return ret_metrics