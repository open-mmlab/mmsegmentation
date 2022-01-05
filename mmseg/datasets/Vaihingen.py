from .builder import DATASETS
from .custom import CustomDataset
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import eval_metrics, pre_eval_to_metrics

@DATASETS.register_module()
class VaihingenDataset(CustomDataset):
    """ISPRS_2d_semantic_labeling_Vaihingen dataset.

    In segmentation map annotation for Vaihingen, 0 stands for the sixth class: Clutter/background
    ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.tif' and ``seg_map_suffix`` is fixed to
    '.tif'.
    """
    CLASSES = (
        'imp surf', 'building', 'low_veg', 'tree', 'car','clutter')

    PALETTE = [[255, 255, 255], [0, 0, 255], 
    [0, 255, 255],[0, 255, 0], [255, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(VaihingenDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='_noBoundary.png',
            reduce_zero_label=True,
            **kwargs)

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            gt_seg_maps = self.get_gt_seg_maps()
            if self.CLASSES is None:
                num_classes = len(
                    reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
            else:
                num_classes = len(self.CLASSES)
            # reset generator
            gt_seg_maps = self.get_gt_seg_maps()
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        aAcc=ret_metrics['aAcc']
        ret_metrics.pop('aAcc', None)
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value[:-1]) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_summary.update({'aAcc':np.round(np.nanmean(aAcc) * 100, 2)})

        # each class table
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results
