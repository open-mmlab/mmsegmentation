# Copyright (c) OpenMMLab. All rights reserved.

import warnings
import numpy as np
from typing import Sequence
from mmengine.logging import print_log
from prettytable import PrettyTable
from mmeval.metrics import MeanIoU

from mmseg.registry import METRICS


@METRICS.register_module()
class IoUMetric(MeanIoU):
    """A wrapper of ``mmeval.MeanIoU``.

    This wrapper implements the `process` method that parses predictions and 
    labels from inputs. This enables ``mmengine.Evaluator`` to handle the data
    flow of different tasks through a unified interface.

    In addition, this wrapper also implements the ``evaluate`` method that
    parses metric results and print pretty tabel of metrics per class.

    Args:
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`mmeval.MeanIoU`.
    """

    def __init__(self, dist_backend='torch_cuda', **kwargs):
        iou_metrics = kwargs.pop('iou_metrics', None)
        if iou_metrics is not None:
            warnings.warn(
                'DeprecationWarning: The `iou_metrics` parameter of '
                '`IoUMetric` is deprecated, defaults return all metrics now!')
        collect_device = kwargs.pop('collect_device', None)

        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`IoUMetric` is deprecated, use `dist_backend` instead.')

        # Changes the default value of `classwise_results` to True.
        super().__init__(classwise_results=True,
                         dist_backend=dist_backend,
                         **kwargs)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        Parse predictions and labels from ``data_samples`` and invoke
        ``self.add``.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        predictions, labels = [], []
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            predictions.append(pred_label)
            labels.append(label)

        self.add(predictions, labels)

    def evaluate(self, *args, **kwargs):
        """Returns metric results and print pretty tabel of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        classwise_results = metric_results['classwise_results']
        del metric_results['classwise_results']
        
        # Pretty table of the metric results per class.
        summary_table = PrettyTable()
        summary_table.add_column('Class', self.dataset_meta['classes'])
        for key, value in classwise_results.items():
            value = np.round(value * 100, 2)
            summary_table.add_column(key, value)

        print_log('per class results:', logger='current')
        print_log('\n' + summary_table.get_string(), logger='current')

        # Multiply value by 100 to convert to percentage and rounding. 
        evaluate_results = {
            k: round(v * 100, 2) for k, v in metric_results.items()}
        return evaluate_results
