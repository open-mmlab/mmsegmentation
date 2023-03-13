# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Sequence

import numpy as np
from mmengine import mkdir_or_exist
from mmengine.dist import is_main_process
from mmengine.logging import print_log
from PIL import Image
from prettytable import PrettyTable

from .iou_metric import IoUMetric

try:
    from mmeval.metrics import MeanIoU
except ImportError:
    MeanIoU = IoUMetric

from mmseg.registry import METRICS


@METRICS.register_module()
class MMEvalIoUMetric(MeanIoU):
    """MeanIoU evaluation metric.

    Args:
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.

    Keyword Args:
        dataset_meta (dict, optional): Meta information of the dataset, this is
            required for some metrics that require dataset information.
            Defaults to None.
        dist_collect_mode (str, optional): The method of concatenating the
            collected synchronization results. This depends on how the
            distributed data is split. Currently only 'unzip' and 'cat' are
            supported. For PyTorch's ``DistributedSampler``, 'unzip' should
            be used. Defaults to 'unzip'.
        dist_backend (str, optional): The name of the distributed communication
            backend, you can get all the backend names through
            ``mmeval.core.list_all_backends()``.
            If None, use the default backend. Defaults to None.
    """

    def __init__(self,
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 **kwargs):

        if isinstance(self, IoUMetric):
            raise TypeError(
                'MMEvalIoUMetric must be a subclass of mmeval.MeanIoU,'
                'please install MMEval first.')

        # Changes the default value of `classwise_results` to True.
        super().__init__(classwise_results=True, **kwargs)

        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

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

            # format_result:
            if self.output_dir:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask += 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

        self.add(predictions, labels)

    def evaluate(self, *args, **kwargs):
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        if self.format_only:
            return {}
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
            k: round(v * 100, 2)
            for k, v in metric_results.items()
        }
        return evaluate_results
