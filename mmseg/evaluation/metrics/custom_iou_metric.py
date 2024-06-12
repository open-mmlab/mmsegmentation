# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
from sklearn.metrics import average_precision_score
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
class CustomIoUMetric(BaseMetric):
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
                 prefix: Optional[str] = "cstm_",
                 thresholds = [0.25, 0.50, 0.75],
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.thresholds = thresholds
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
        # print(f"data samples: \n {data_samples}")
        # print(f"data_batch:\n {data_batch}")
       
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                
                self.results.append(
                    self.intersect_and_union(pred_label, label, num_classes,
                                             self.ignore_index))
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
        assert len(results) == 4
    
        ret_metrics = self.total_area_to_metrics(
            area_intersect=results[0], area_union=results[1],
            area_pred_label=results[2], area_label=results[3],
            thresholds=self.thresholds, metrics=self.metrics
        )
            
            # results[0], results[1],results[2],results[3], 
            # self.metrics, self.nan_to_num, self.beta
        
        
        
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
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
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
        print(f"pred_label: {[pred_label.shape]}")
        print(f"label: {label.shape}")
        mask = (label != ignore_index)
        print(f"mask: {np.unique(mask.cpu())}")
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        print('#' * 40)
        print(f"pred_label: {np.unique(pred_label.cpu())}")
        print(f"label: {np.unique(label.cpu())}")
        print(f"intersect: {np.unique(intersect.cpu())}")
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
    def map_metrics(
        area_intersect: tuple,
        area_union: tuple,
        area_pred_label: tuple,
        area_label: tuple,
        thresholds: list = [0.25, 0.50, 0.75]      
    ):
        """_summary_

        Args:
            area_intersect (tuple): n_inputs x n_classes, tuple, tensor, tensor
            area_union (tuple): n_inputs x n_classes
            area_pred_label (tuple): n_inputs x n_classes
            area_label (tuple): n_inputs x n_classes
        """
        # Generate prediction scores

        
        
        n_inputs = len(area_intersect)
        n_classes = len(area_intersect[0])
        data = dict()
        for threshold in thresholds:
            data["ap" + str(int(threshold * 100))] = []
        data["aIoU"] = []
        for class_idx in range(n_classes):
            class_iou = []
            class_y_true = []
            print(f"\nClass {class_idx}:")
            for input_idx in range(n_inputs):
                # get iou
                print(f"area intersect: {area_intersect[input_idx][class_idx]}")
                print(f"area_union: {area_union[input_idx][class_idx]}")
                print(f"area_label: {area_label[input_idx][class_idx]}")
                print(f"area_pred_label: {area_pred_label[input_idx][class_idx]}")
                iou = area_intersect[input_idx][class_idx] / area_union[input_idx][class_idx]
                # print(f"local iou: {iou}")
                if area_union[input_idx][class_idx] == 0 or area_intersect[input_idx][class_idx] == 0:
                    iou = 0
                class_iou.append(iou)
                y_true = 1 if area_label[input_idx][class_idx] > 0 else 0
                class_y_true.append(y_true)
            print(f"iou class : {class_iou}")
            
            data["aIoU"].append(sum(class_iou) / sum(class_y_true))
            for threshold in thresholds:
                print(f"threshold: {threshold}")
                y_pred = [1 if score >= threshold else 0 for score in class_iou]
                # AP score now is interpreted as binary task...
                ap_score = average_precision_score(y_true=class_y_true, y_score=y_pred)
                print(f"y_pred: {y_pred}")
                print(f"y_tue_class: {class_y_true}")
                print(f"ap score: {ap_score}")
                ap_name = "ap" + str(int(threshold * 100))   
                data[ap_name].append(ap_score)
            
        return data
                
            
        # for class in n_classes:
        #   get iou over 11 examples
        #   get 
        
        # class_map25 = [1 if iou_score >= 0.25 else 0 for iou_score in iou_per_class]
        # class_map50 = [1 if iou_score >= 0.50 else 0 for iou_score in iou_per_class]
        # class_map75 = [1 if iou_score >= 0.75 else 0 for iou_score in iou_per_class]
        
        


    @staticmethod
    def total_area_to_metrics(
        area_intersect: tuple,
        area_union: tuple,
        area_pred_label: tuple,
        area_label: tuple,
        thresholds: list = [0.25, 0.50, 0.75],  
        metrics: List[str] = ['mIoU'],
        nan_to_num: Optional[int] = None,
        beta: int = 1
        ):
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
        total_area_intersect = sum(area_intersect)
        total_area_union = sum(area_union)
        total_area_pred_label = sum(area_pred_label)
        total_area_label = sum(area_label)
          
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
                print(f"presision: {precision.shape}\nrecall: {recall.shape}")
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall
            elif metric == 'mAP':
                ap_data = CustomIoUMetric.map_metrics(
                    area_intersect=area_intersect,
                    area_union=area_union,
                    area_pred_label=area_pred_label,
                    area_label=area_label,
                    thresholds=thresholds
                )
                for key, value in ap_data.items():
                    ret_metrics[key] = torch.tensor(value)
                
                    
                    
                   

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
