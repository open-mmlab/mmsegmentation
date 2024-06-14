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
import copy
import warnings
from collections import OrderedDict
from typing import List, Optional, Sequence, Union
from math import sqrt
import numpy as np
from datetime import datetime

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
        print(f"process at time: {datetime.now().strftime('%H:%M:%S')}")
        num_classes = len(self.dataset_meta['classes'])
       
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            
            prediction_probs = np.zeros(pred_label.shape)
            print(f"process: pre convert sigmoid at time: {datetime.now().strftime('%H:%M:%S')}")
            pixel_probs = torch.sigmoid(data_sample['seg_logits']['data'])
            print(f"process: converted sigmoid at time: {datetime.now().strftime('%H:%M:%S')}")
            for row in range(pred_label.shape[0]):
                for col in range(pred_label.shape[1]):
                    # prob of pred label
                    prediction_probs[row][col] = pixel_probs[pred_label[row][col]][row][col]
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                self.results.append(
                    CustomIoUMetric.extract_relevant_data(
                        pred_label=pred_label, label=label,
                        num_classes=num_classes, probs=prediction_probs
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
        assert len(results) == 3
    
        ret_metrics = self.map_metrics(
            predicted_clusters=results[0],
            ground_truth_clusters=results[1],
            probs=results[2]
        )
        
        
        
        
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
    def extract_relevant_data(
        pred_label: torch.tensor, label: torch.tensor,
        num_classes: int,  probs: torch.tensor
    ):
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
        print(f"extract relelvant data at time: {datetime.now().strftime('%H:%M:%S')}")
        predicted_clusters = []
        ground_truth_clusters = []
        for class_idx in range(num_classes):
            print(f"get pred class clusters of class {class_idx} at time: {datetime.now().strftime('%H:%M:%S')}")
            predicted_clusters.append(
                Clusters.get_clusters_class(
                    seg_map=pred_label.cpu().numpy(),
                    class_idx=class_idx
                )
            )
            print(f"finished pred class clusters of class {class_idx} at time: {datetime.now().strftime('%H:%M:%S')}")
            ground_truth_clusters.append(
                Clusters.get_clusters_class(
                    seg_map=label.cpu().numpy(),
                    class_idx=class_idx
                )
            )
            print(f"get gt class clusters of class {class_idx} at time: {datetime.now().strftime('%H:%M:%S')}")
        return predicted_clusters, ground_truth_clusters, probs
        
        
        # mask = (label != ignore_index)
        # pred_label = pred_label[mask]
        # label = label[mask]
         
        

        # intersect = pred_label[pred_label == label]
        # area_intersect = torch.histc(
        #     intersect.float(), bins=(num_classes), min=0,
        #     max=num_classes - 1).cpu()
        # area_pred_label = torch.histc(
        #     pred_label.float(), bins=(num_classes), min=0,
        #     max=num_classes - 1).cpu()
        # area_label = torch.histc(
        #     label.float(), bins=(num_classes), min=0,
        #     max=num_classes - 1).cpu()
        # area_union = area_pred_label + area_label - area_intersect
        
        # return area_intersect, area_union, area_pred_label, area_label
    
    @staticmethod
    def map_metrics(
        predicted_clusters: tuple, ground_truth_clusters: tuple,
        probs: tuple,
        thresholds: list = [0.25, 0.50, 0.75]      
    ) -> dict:
        """_summary_

        Args:
            predicted_clusters (tuple): (n_inputs x n_classes)
            ground_truth_clusters (tuple): _description_
            probs (tuple): _description_
            thresholds (list, optional): _description_. Defaults to [0.25, 0.50, 0.75].

        Returns:
            _type_: _description_
        """
        print(f"map_metrics at time: {datetime.now().strftime('%H:%M:%S')}")
        n_samples = len(predicted_clusters)
        n_classes = len(predicted_clusters[0])
        data = {}
        for iou_threshold in thresholds:
            data["ap" + str(int(iou_threshold * 100))] = []
        for iou_threshold in thresholds:
            print(f"map_metrics: iou_th {iou_threshold} at time: {datetime.now().strftime('%H:%M:%S')}")
            # get predicted instances and match with gt 
            
            for class_idx in range(n_classes):
                class_record = []
                # detection_record = {
                #     # "pred_cluster"      :   [],
                #     # "pred_score"        :   [],
                #     # "IoU"               :   [],
                #     # "confusion_label"   :   []
                # }
                total_GTs_class = 0
                for sample_idx in range(n_samples): 
                    detection_record = {}
                    clusters_pred = predicted_clusters[sample_idx][class_idx]
                    clusters_GT = ground_truth_clusters[sample_idx][class_idx]
                    total_GTs_class += len(clusters_GT)
                    
                    for pred_clust in clusters_pred:
                        detection_record['pred_cluster'] = pred_clust
                        pred_score = Clusters.get_av_prob_cluster(
                                cluster=pred_clust,
                                probs=probs[sample_idx]
                        )
                        detection_record['pred_score'] = pred_score
                        # check for: if there is intersect, or gt of class
                        if not clusters_GT:
                            detection_record['confusion_label'] = 'FP'
                            detection_record['IoU'] = 0
                            class_record.append(detection_record)
                            continue
                        closest_GT = Clusters.get_closest_gt(
                            pred_cluster=pred_clust,
                            class_gt_clusters=clusters_GT
                        )
                        area_intersect = Clusters.area_intersection_clusters(
                            cluster0=pred_clust,
                            cluster1=closest_GT
                        )
                        if area_intersect < 1:
                            detection_record['confusion_label'] = 'FP'
                            detection_record['IoU'] = 0
                            class_record.append(detection_record)
                            continue
                        
                        IoU = Clusters.IoU(
                            cluster0=pred_clust,
                            cluster1=closest_GT
                        )
                        
                        detection_record['IoU'] = IoU
                        detection_record["confusion_label"] = CustomIoUMetric.get_confusion_label(
                                iou=IoU,
                                iou_threshold=iou_threshold
                        )
                        class_record.append(detection_record)
                    
                        
                # collected the data for all the samples and each detection
                # Data is sorted in descending order for conf thresholds
                sorted_class_record = sorted(
                    class_record, 
                    key=lambda d: d["pred_score"],
                    reverse=True
                )
                print(f"len sorted class record: {len(sorted_class_record)}")  
                precision, recall = CustomIoUMetric.get_PR_curve(
                    sorted_class_record=sorted_class_record,
                    total_GTs_class=total_GTs_class
                )
                class_ap = CustomIoUMetric.get_average_precision(
                    precision=precision,
                    recall=recall
                )
                data["ap" + str(int(iou_threshold * 100))].append(class_ap)
            print(f"map_metrics: finished iou_th {iou_threshold} at time: {datetime.now().strftime('%H:%M:%S')}")   
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
    def get_PR_curve(sorted_class_record: list, total_GTs_class: int) -> tuple:
        precision = []
        recall = []
        confusion_labels = {
            "TP"    :   0,
            "FP"    :   0
        }
        for prediction in sorted_class_record:
            confusion_labels[prediction["confusion_label"]] += 1
            precision.append(
                CustomIoUMetric.get_precision_numeric(
                    TP=confusion_labels["TP"],
                    FP=confusion_labels["FP"]
                )
            )
            recall.append(
                CustomIoUMetric.get_recall_numeric(
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
    
    

class Clusters:
    @staticmethod
    def get_closest_gt(pred_cluster: list, class_gt_clusters: list) -> list:
        dist_min = Clusters.distance_clusters(
                    cluster0=pred_cluster,
                    cluster1=class_gt_clusters[0],
                    mode='center'
        )
        closest_clust = class_gt_clusters[0]
        if len(class_gt_clusters) == 1:
            return closest_clust
        for gt_cluster in class_gt_clusters[1:]:
            dist = Clusters.distance_clusters(
                    cluster0=pred_cluster,
                    cluster1=gt_cluster,
                    mode='center'
            )
            if dist < dist_min:
                closest_clust = gt_cluster
                dist_min = dist
        return closest_clust
    
    @staticmethod
    def get_av_prob_cluster(cluster: list, probs: np.ndarray) -> float:
        total_prob = 0
        for pixel in cluster:
            total_prob += probs[pixel[0]][pixel[1]]
        return total_prob / len(cluster)
    
    @staticmethod
    def intersection_clusters(cluster0: list, cluster1: list) -> list:
        intersection = []
        for pixel_0 in cluster0:
            for pixel_1 in cluster1:
                if pixel_0 == pixel_1:
                    intersection.append(pixel_1)
        return intersection
    
    @staticmethod
    def area_intersection_clusters(cluster0: list, cluster1: list) -> int:
        area_intersection = 0
        for pixel_0 in cluster0:
            for pixel_1 in cluster1:
                if pixel_0 == pixel_1:
                    area_intersection += 1
        return area_intersection
    
    @staticmethod
    def IoU(cluster0: list, cluster1: list) -> float:
        area_intersect = Clusters.area_intersection_clusters(
            cluster0 = cluster0,
            cluster1 = cluster1
        )
        area_union = Clusters.area_union_clusters(
            cluster0 = cluster0,
            cluster1 = cluster1
        )
        if area_union == 0:
            return 0
        return area_intersect / area_union
    
    @staticmethod
    def area_union_clusters(cluster0: list, cluster1: list) -> int:
        area_intersect = Clusters.area_intersection_clusters(
            cluster0=cluster0, cluster1=cluster1
        )
        return len(cluster0) + len(cluster1) - area_intersect
    @staticmethod
    def distance_clusters(cluster0: list, cluster1: list, mode: str = 'center') -> float:
        if mode == 'center':
            center0 = Clusters.get_center_cluster(cluster=cluster0)
            center1 = Clusters.get_center_cluster(cluster=cluster1)
            return Clusters.euclidean_dist(point0=center0, point1=center1)
        print("distance cluster not yet implemented")
    
    @staticmethod
    def get_center_cluster(cluster: list) -> tuple:
        row_center = 0
        col_center = 0
        for pixel in cluster:
            row_center += pixel[0]
            col_center += pixel[1]
        return (int(row_center / len(cluster)), int(col_center / len(cluster)))
            
    
    @staticmethod
    def get_closest_points_clusters(cluster0: list, cluster1: list) -> tuple:
        pass 
    
    @staticmethod
    def euclidean_dist(point0, point1) -> float:
        return sqrt((point0[0] - point1[0]) ** 2 + (point0[1] - point1[1]) ** 2)
    
    
    # pred must be deepcopy and to cpu/numpy
    @staticmethod
    def get_clusters_class(
        seg_map: np.ndarray, class_idx: int, 
        min_cluster_size: int = 200
    ):
        visited = Clusters.make_visited_arr(seg_map=seg_map, class_idx=class_idx)
        clusters = []
        while np.sum(visited) < visited.shape[0] * visited.shape[1]:
            class_pixel = Clusters.find_class_pixel(visited=visited)
            if class_pixel:
                cluster = Clusters.BFS_find_cluster(
                    pixel=class_pixel, visited=visited
                )
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)
        return clusters       
    # pixel is (row, col)
    @staticmethod
    def BFS_find_cluster(pixel: tuple, visited: np.ndarray):
        cluster = []
        queue = list()
        # visited[pixel[0]][pixel[1]] = True
        # queue.append(pixel)
        # cluster.append(pixel)
        Clusters.add_pixel_to_cluster(
            pixel=pixel, visited=visited,
            cluster=cluster, queue=queue
        )
        while queue:
            pixel = queue.pop(0)
            pixels = Clusters.expand_pixel(pixel=pixel)
            for pixel in pixels:
                if not Clusters.pixel_in_bounds(
                    pixel=pixel, height=visited.shape[0], 
                    width=visited.shape[1]
                ):
                    continue
                if visited[pixel[0]][pixel[1]]:
                    continue
                Clusters.add_pixel_to_cluster(
                    pixel=pixel, visited=visited,
                    cluster=cluster,
                    queue=queue
                )
        return cluster
            
    
    @staticmethod
    def add_pixel_to_cluster(
        pixel: tuple, visited: np.ndarray, 
        cluster: list, queue: list
    ):
        visited[pixel[0]][pixel[1]] = True
        queue.append(pixel)
        cluster.append(pixel)
        # return visited, queue, cluster
            
    # Expand pixel
    @staticmethod
    def expand_pixel(pixel: tuple) -> list:
        return [(pixel[0] + row, pixel[1] + col) for row in [-1, 0, 1] for col in [-1, 0, 1] if not (row == 0 and col == 0)]
    
    @staticmethod
    def pixel_in_bounds(pixel: tuple, height: int, width: int):
        return 0 <= pixel[0] < height and 0 <= pixel[1] < width
    
    @staticmethod
    def find_class_pixel(visited: np.ndarray) -> tuple:
        for row in range(visited.shape[0]):
            for col in range(visited.shape[1]):
                if not visited[row][col]:
                    return (row, col)
        return None

    @staticmethod
    def make_visited_arr(seg_map: np.ndarray, class_idx: int):
        return (seg_map != class_idx)
    
    
                

                
            
            