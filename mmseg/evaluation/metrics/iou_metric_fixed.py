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
class IoUMetricFixed(BaseMetric):
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
                 iou_metrics: List[str] = ['mIoU', 'mDice', 'mFscore', 'mAP'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = "_",
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
            prediction_probs = torch.zeros(pred_label.shape)
            pixel_probs = torch.sigmoid(data_sample['seg_logits']['data'])
            for row in range(pred_label.shape[0]):
                for col in range(pred_label.shape[1]):
                    # prob of pred label
                    prediction_probs[row][col] = pixel_probs[pred_label[row][col]][row][col]
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                self.results.append(
                    self.get_relevant_data(
                        pred_label=pred_label.cpu(),
                        label=label.cpu(),
                        prediction_probs=prediction_probs.cpu(),
                        num_classes=num_classes,
                        ignore_index=None
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
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 5

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)
        data = IoUMetricFixed.get_AP_metric(
            area_intersect=results[0],
            area_union=results[1],
            area_pred_label=results[2],
            area_label=results[3],
            av_class_probs=results[4]
        )
       
        for key, val in data.items():
            ret_metrics[key] = val
       
        class_names = self.dataset_meta['classes']

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
    def get_relevant_data(
        pred_label: torch.tensor, label: torch.tensor,
        prediction_probs: torch.tensor,
        num_classes: int, ignore_index: int):
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

        # mask = (label != ignore_index)
        # pred_label = pred_label[mask]
        # label = label[mask]

        intersect = pred_label[pred_label == label]
        av_class_probs = []
        for class_idx in range(num_classes):
            class_probs = prediction_probs[pred_label == class_idx]
            if len(class_probs) == 0:
                av_class_probs.append(0)
                continue
            av_class_probs.append(
                sum(class_probs) / len(class_probs)
            )
        av_class_probs = torch.tensor(av_class_probs)
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        # label = label[mask] # TODO not sure
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label, av_class_probs

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU', 'mDice', 'mFscore'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
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

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'mAP']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
    
    @staticmethod
    def get_AP_metric(
            area_intersect,
            area_union,
            area_pred_label,
            area_label,
            av_class_probs,
            iou_thresholds = [0.25, 0.5, 0.75]
    ) -> dict:
        n_samples = len(area_intersect)
        n_classes = len(area_intersect[0])
        data = {}
        for iou_threshold in iou_thresholds:
            data["ap" + str(int(iou_threshold * 100))] = []
        for iou_threshold in iou_thresholds:
            
            # get predicted instances and match with gt 
            
            for class_idx in range(n_classes):
                class_record = []
                total_gts_class = 0
                for sample_idx in range(n_samples): 
                    detection_record = {}
                    pred_score = av_class_probs[sample_idx][class_idx]
                    iou = IoUMetricFixed.get_iou(
                        area_intersect=area_intersect[sample_idx][class_idx],
                        area_union=area_union[sample_idx][class_idx]
                    )
                    if area_label[sample_idx][class_idx] > 0:
                        total_gts_class += 1
                    detection_record["pred_score"] = pred_score
                    detection_record["IoU"] = iou
                    detection_record["confusion_label"] = IoUMetricFixed.get_confusion_label(iou=iou, iou_threshold=iou_threshold)
                    detection_record["area_intersect"] = area_intersect[sample_idx][class_idx]
                    detection_record["area_gt"] = area_label[sample_idx][class_idx]
                    detection_record["area_pred"] = area_pred_label[sample_idx][class_idx]
                    for key, val in detection_record.items():
                        if key != "confusion_label" and val < 0:
                            print(f"negative val: {val} at key {key}")
                    class_record.append(detection_record)  
                # Data is sorted in descending order for conf thresholds
                sorted_class_record = sorted(
                    class_record, 
                    key=lambda d: d["pred_score"],
                    reverse=True
                )
                # precision, recall = IoUMetricFixed.get_PR_curve_area(
                #     sorted_class_record=sorted_class_record
                # ) 
                precision, recall = IoUMetricFixed.get_PR_curve_numeric(
                    sorted_class_record=sorted_class_record, 
                    total_GTs_class=total_gts_class
                ) 
                
                pr_dict_list = [{"P" : p, "R" : r} for p, r in zip(precision, recall)]
                sorted_PR = sorted(
                    pr_dict_list,
                    key=lambda d: d["R"],
                    reverse=True
                )
                precision = [item["P"] for item in sorted_PR]
                recall = [item["R"] for item in sorted_PR]
                class_ap = IoUMetricFixed.get_average_precision(
                    precision=precision,
                    recall=recall
                )
                data["ap" + str(int(iou_threshold * 100))].append(class_ap)
        for iou_threshold in iou_thresholds:
            data["ap" + str(int(iou_threshold * 100))] = np.asarray(
                [
                    float(value) for value in
                    data["ap" + str(int(iou_threshold * 100))]
                ]
            )
        return data
    @staticmethod
    def get_average_precision(precision: list, recall: list):
        
        recall.append(0)
        precision.append(1)
        n_points = len(precision)
        return sum(
            [
                (recall[k] - recall[k + 1]) * precision[k] 
                    for k in range(n_points - 1)
            ]
        )
    @staticmethod
    def get_PR_curve_area(sorted_class_record: list) -> tuple:
        precision = []
        recall = []
        for prediction in sorted_class_record:
            
            precision.append(
                IoUMetricFixed.get_precision_area(
                    area_intersect=prediction["area_intersect"],
                    area_pred=prediction["area_pred"]
                )
            )
            recall.append(
                IoUMetricFixed.get_recall_area(
                    area_intersection=prediction["area_intersect"],
                    area_gt=prediction["area_gt"]
                )
            )
        return precision, recall
    @staticmethod
    def get_PR_curve_numeric(sorted_class_record: list, total_GTs_class: int) -> tuple:
        precision = []
        recall = []
        confusion_labels = {
            "TP"    :   0,
            "FP"    :   0
        }
        for prediction in sorted_class_record:
            confusion_labels[prediction["confusion_label"]] += 1
            precision.append(
                IoUMetricFixed.get_precision_numeric(
                    TP=confusion_labels["TP"],
                    FP=confusion_labels["FP"]
                )
            )
            recall.append(
                IoUMetricFixed.get_recall_numeric(
                    TP=confusion_labels["TP"],
                    total_GT=total_GTs_class
                )
            )
        return precision, recall
        
    @staticmethod
    def get_recall_numeric(TP: int, total_GT: int) -> float:
        if total_GT == 0:
            return 0
        return TP / total_GT
    
    @staticmethod   
    def get_recall_area(area_intersection: int, area_gt: int) -> float:
        if area_gt == 0:
            return 0
        return area_intersection / area_gt
    
    @staticmethod
    def get_precision_numeric(TP: int, FP: int) -> float:
        if TP + FP == 0:
            return 0
        return TP / (TP + FP)
    
    @staticmethod
    def get_precision_area(area_intersect: int, area_pred: int) -> float:
        if area_pred == 0:
            return 0
        return area_intersect / area_pred
    
    @staticmethod
    def get_confusion_label(iou, iou_threshold):
        if iou >= iou_threshold:
            return 'TP'
        else:
            return 'FP'
        
    @staticmethod
    def get_iou(area_intersect, area_union):
        if area_union == 0:
            return 0
        return area_intersect / area_union