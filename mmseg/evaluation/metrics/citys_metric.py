# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist, scandir
from PIL import Image

from mmseg.registry import METRICS


@METRICS.register_module()
class CitysMetric(BaseMetric):
    """Cityscapes evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        citys_metrics (list[str] | str): Metrics to be evaluated,
            Default: ['cityscapes'].
        to_label_id (bool): whether convert output to label_id for
            submission. Default: True.
        suffix (str): The filename prefix of the png files.
            If the prefix is "somepath/xxx", the png files will be
            named "somepath/xxx.png". Default: '.format_cityscapes'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 citys_metrics: List[str] = ['cityscapes'],
                 to_label_id: bool = True,
                 suffix: str = '.format_cityscapes',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = citys_metrics
        assert self.metrics[0] == 'cityscapes'
        self.to_label_id = to_label_id
        self.suffix = suffix

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        mkdir_or_exist(self.suffix)

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'][0].cpu().numpy()
            # results2img
            if self.to_label_id:
                pred_label = self._convert_to_label_id(pred_label)
            basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
            png_filename = osp.join(self.suffix, f'{basename}.png')
            output = Image.fromarray(pred_label.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color
            output.putpalette(palette)
            output.save(png_filename)

        ann_dir = osp.join(data_samples[0]['seg_map_path'].split('val')[0],
                           'val')
        self.results.append(ann_dir)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): Testing results of the dataset.

        Returns:
            dict[str: float]: Cityscapes evaluation results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        try:
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install cityscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'

        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_dir = self.suffix

        eval_results = dict()
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        CSEval.args.evalInstLevelScore = True
        CSEval.args.predictionPath = osp.abspath(result_dir)
        CSEval.args.evalPixelAccuracy = True
        CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []
        ann_dir = results[0]
        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        for seg_map in scandir(ann_dir, 'gtFine_labelIds.png', recursive=True):
            seg_map_list.append(osp.join(ann_dir, seg_map))
            pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))
        metric = dict()
        eval_results.update(
            CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))
        metric['averageScoreCategories'] = eval_results[
            'averageScoreCategories']
        metric['averageScoreInstCategories'] = eval_results[
            'averageScoreInstCategories']
        return metric

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy
