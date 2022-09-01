# Copyright (c) OpenMMLab. All rights reserved.

import warnings
import numpy as np
from typing import Sequence
from mmengine.logging import print_log
from prettytable import PrettyTable
from mmeval.segmentation import MeanIoU

from mmseg.registry import METRICS


@METRICS.register_module()
class IoUMetric(MeanIoU):
    """A wrapper of ``mmeval.MeanIoU``.

    This wrapper implements the `process` method that parses predictions and 
    labels from inputs. This enables ``mmengine.Evaluator`` to handle the data
    flow of different tasks through a unified interface.

    In addition, this wrapper also implements the `evaluate` method that parses 
    metric results and print pretty tabel of metrics per class.
    """

    def __init__(self, verbose_results=True, **kwargs):
        """Changes the default value of `verbose_results` to True."""
        iou_metrics = kwargs.pop('iou_metrics', None)
        if iou_metrics is not None:
            warnings.warn(
                'DeprecationWarning: The `iou_metrics` parameter of '
                '`IoUMetric` is deprecated, defaults return all metrics now!')
        super().__init__(verbose_results=verbose_results, **kwargs)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        Parse predictions and labels from `data_samples` and invoke `self.add`.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            self.add(pred_label, label)

    def evaluate(self, *args, **kwargs):
        """Returns metric results and print pretty tabel of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)

        # We only return the averaged metric results.
        evaluate_results = dict()
        # Pretty table of the metric results per class.
        summary_table = PrettyTable()

        summary_table.add_column('Class', self.dataset_meta['classes'])
        for key, value in metric_results.items():
            # Multiply value by 100 to convert to percentage and rounding. 
            if key.startswith('m') or key == 'aAcc':
                # Add averaged metric results to `evaluate_results`
                evaluate_results[key] = round(value * 100, 2)
            else:
                value = np.round(value.cpu().numpy() * 100, 2)
                summary_table.add_column(key, value)

        print_log('per class results:', logger='current')
        print_log('\n' + summary_table.get_string(), logger='current')
        return evaluate_results
