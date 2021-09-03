# Copyright (c) OpenMMLab. All rights reserved.
import bisect
from itertools import chain

import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    support evaluation and formatting results

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the concatenated
            dataset results separately, Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE
        self.separate_eval = separate_eval
        assert separate_eval in [True, False], \
            f'separate_eval can only be True or False,' \
            f'but get {separate_eval}'
        if any([isinstance(ds, CityscapesDataset) for ds in datasets]):
            raise NotImplementedError(
                'Evaluating ConcatDataset containing CityscapesDataset'
                'is not supported!')

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]]): per image
                pre_eval results or predict segmentation map for
                computing evaluation metric.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: evaluate results of the total dataset
                or each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset in zip(self.cumulative_sizes, self.datasets):
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'\nEvaluateing {dataset.img_dir} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results

        if len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types when '
                'self.separate_eval=False')
        else:
            if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                    results, str):
                # merge the generators of gt_seg_maps
                gt_seg_maps = chain(
                    *[dataset.get_gt_seg_maps() for dataset in self.datasets])
            else:
                # if the results are `pre_eval` results,
                # we do not need gt_seg_maps to evaluate
                gt_seg_maps = None
            eval_results = self.datasets[0].evaluate(
                results, gt_seg_maps=gt_seg_maps, logger=logger, **kwargs)
            return eval_results

    def get_dataset_idx_and_sample_idx(self, indice):
        """Return dataset and sample index when given an indice of
        ConcatDataset.

        Args:
            indice (int): indice of sample in ConcatDataset

        Returns:
            int: the index of sub dataset the sample belong to
            int: the index of sample in its corresponding subset
        """
        if indice < 0:
            if -indice > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            indice = len(self) + indice
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, indice)
        if dataset_idx == 0:
            sample_idx = indice
        else:
            sample_idx = indice - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """format result for every sample of ConcatDataset."""
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        ret_res = []
        for i, indice in enumerate(indices):
            dataset_idx, sample_idx = self.get_dataset_idx_and_sample_idx(
                indice)
            res = self.datasets[dataset_idx].format_results(
                [results[i]],
                imgfile_prefix + f'/{dataset_idx}',
                indices=[sample_idx],
                **kwargs)
            ret_res.append(res)
        return sum(ret_res, [])

    def pre_eval(self, preds, indices):
        """do pre eval for every sample of ConcatDataset."""
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]
        ret_res = []
        for i, indice in enumerate(indices):
            dataset_idx, sample_idx = self.get_dataset_idx_and_sample_idx(
                indice)
            res = self.datasets[dataset_idx].pre_eval(preds[i], sample_idx)
            ret_res.append(res)
        return sum(ret_res, [])


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item from original dataset."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``"""
        return self.times * self._ori_len
