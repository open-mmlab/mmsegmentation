# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations
import torch.nn.functional as F
import torch
from torchmetrics.functional import calibration_error
from ..utils import brierscore


@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
        gt_seg_map_loader_cfg (dict, optional): build LoadAnnotations to
            load gt for evaluation, load from disk by default. Default: None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend='disk')):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        raise NotImplementedError

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return results['gt_semantic_seg']

    def get_gt_seg_map_by_idx_and_reduce_zero_label(self, index):
        seg_gt = self.get_gt_seg_map_by_idx(index)
        if self.reduce_zero_label:
            seg_gt[seg_gt == 0] = 255
            seg_gt = seg_gt - 1
            seg_gt[seg_gt == 254] = 255
        return seg_gt

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']

    def pre_eval_custom(self, seg_logit, seg_gt, logit2prob="softmax", use_bags=False, bags_kwargs={}):
        seg_logit = seg_logit.cpu()
        num_cls = seg_logit.shape[1]

        seg_gt_tensor_flat = torch.from_numpy(seg_gt).type(torch.long).flatten()  # [W, H] => [WxH]
        seg_logit_flat = seg_logit.flatten(2, -1).squeeze().permute(1, 0)  # [1, K, W, H] => [WxH, K]
        if self.ignore_index:
            ignore_bg_mask = (seg_gt_tensor_flat == self.ignore_index)  # ignore bg pixels
        else:
            ignore_bg_mask = torch.zeros_like(seg_gt_tensor_flat)

        if not use_bags:
            if logit2prob == "edl":
                alpha = seg_logit_flat + 1
                strength = alpha.sum(dim=1, keepdim=True)
                u = num_cls / strength
                probs = alpha / strength
                seg_max_prob = 1 - u
                # seg_max_prob = probs.max(dim=1)[0]
                seg_max_logit = seg_logit_flat.max(dim=1)[0]
                seg_entropy = - (probs * probs.log()).sum(1)
                seg_entropy = (torch.lgamma(alpha).sum(1, keepdim=True) - torch.lgamma(strength) -
                               (num_cls - strength) * torch.digamma(strength) -
                               ((alpha - 1.0) * torch.digamma(alpha)).sum(1, keepdim=True))

            else:
                probs = F.softmax(seg_logit_flat, dim=1)
                seg_max_prob = probs.max(dim=1)[0]
                seg_max_logit = seg_logit_flat.max(dim=1)[0]
                seg_entropy = - (probs * probs.log()).sum(1)
            # seg_pred_all_other = torch.zeros_like(seg_logit_flat, dtype=torch.bool)
        else:
            assert 'num_bags' in bags_kwargs and 'bag_masks' in bags_kwargs
            probs = torch.zeros_like(seg_logit_flat[:, :-bags_kwargs["num_bags"]])
            logits = torch.zeros_like(seg_logit_flat[:, :-bags_kwargs["num_bags"]])
            is_other = torch.zeros_like(seg_logit_flat[:, :bags_kwargs["num_bags"]])
            seg_pred_entropy = 0
            for bag_idx in range(bags_kwargs["num_bags"]):
                cp_seg_logit = seg_logit_flat.clone()
                bag_seg_logit = cp_seg_logit[:, bags_kwargs["bag_masks"][bag_idx]]
                # ignores other after softmax
                bag_probs = F.softmax(bag_seg_logit, dim=1)
                oth_index = bag_probs.shape[1] - 1
                is_other[:, bag_idx] = (bag_probs.argmax(dim=1) == oth_index)
                seg_pred_entropy += -(bag_probs * bag_probs.log()).sum(1)
                bag_smp_no_other = bag_probs[:, :oth_index]
                bag_seg_logit_no_other = bag_seg_logit[:, :oth_index]
                mask = bags_kwargs["bag_masks"][bag_idx][:-bags_kwargs["num_bags"]]
                probs[:, mask] = bag_smp_no_other
                logits[:, mask] = bag_seg_logit_no_other
            seg_max_prob = probs.max(dim=1)[0]
            seg_max_logit = logits.max(dim=1)[0]
            seg_entropy = seg_pred_entropy / bags_kwargs["num_bags"]
            # seg_pred_all_other = is_other.all(1, True)
        # Compute OOD metrics for openset
        if hasattr(self, "ood_indices"):
            ood_mask = (seg_gt_tensor_flat == self.ood_indices[0])
            ood_valid = ood_mask.any() and (~ood_mask).any()
            if ood_valid:
                out_scores_probs, in_scores_probs = self.get_in_out_conf(seg_max_prob.cpu().numpy(), seg_gt_tensor_flat.cpu().numpy(), "max_prob")
                auroc_prob, aupr_prob, fpr_prob = self.evaluate_ood(out_scores_probs, in_scores_probs)
                probs_ood = np.array([auroc_prob, aupr_prob, fpr_prob])

                out_scores_logit, in_scores_logit = self.get_in_out_conf(seg_max_logit.cpu().numpy(), seg_gt_tensor_flat.cpu().numpy(), "max_logit")
                auroc_logit, aupr_logit, fpr_logit = self.evaluate_ood(out_scores_logit, in_scores_logit)
                logit_ood = np.array([auroc_logit, aupr_logit, fpr_logit])

                out_scores_entropy, in_scores_entropy = self.get_in_out_conf(seg_entropy.cpu().numpy(), seg_gt_tensor_flat.cpu().numpy(), "entropy")
                auroc_entropy, aupr_entropy, fpr_entropy = self.evaluate_ood(out_scores_entropy, in_scores_entropy)
                entropy_ood = np.array([auroc_entropy, aupr_entropy, fpr_entropy])
                ood_metrics = (np.hstack((probs_ood, logit_ood, entropy_ood)), True)
            else:
                ood_metrics = (np.array([0. for _ in range(9)]), True)
        else:
            # Puts nans otherwise
            ood_metrics = (np.array([0. for _ in range(9)]), False)

        # Calibration/Confidence metrics for closed set
        if not hasattr(self, "ood_indices"):
            if self.ignore_index:
                seg_gt_tensor_flat_no_bg = seg_gt_tensor_flat[~ignore_bg_mask]
                probs_no_bg = probs[~ignore_bg_mask, :]

            nll = F.nll_loss(probs_no_bg.log(), seg_gt_tensor_flat_no_bg, reduction='mean').item()
            ece_l1 = calibration_error(probs_no_bg, seg_gt_tensor_flat_no_bg, norm='l1', ).item()
            ece_l2 = calibration_error(probs_no_bg, seg_gt_tensor_flat_no_bg, norm='l2').item()
            brier = brierscore(probs_no_bg, seg_gt_tensor_flat_no_bg, reduction="mean").item()
            calib_metrics = np.array([nll, ece_l1, ece_l2, brier])

            per_cls_prob = [0. for _ in range(num_cls)]
            per_cls_u = [0. for _ in range(num_cls)]
            per_cls_strength = [0. for _ in range(num_cls)]

            for c in range(num_cls):
                mask_cls = (seg_gt_tensor_flat == c)
                if mask_cls.any():
                    per_cls_prob[c] = probs[mask_cls, c].sum().item()
                    if logit2prob == "edl":
                        per_cls_u[c] = u[mask_cls].sum().item()
                        per_cls_strength[c] = strength[mask_cls].sum().item()

            per_cls_prob = np.array(per_cls_prob)
            per_cls_u = np.array(per_cls_u)
            per_cls_strength = np.array(per_cls_strength)
            per_cls_conf_metrics = (per_cls_prob, per_cls_u, per_cls_strength)
        else:
            calib_metrics = np.array([0 for _ in range(4)])
            per_cls_prob = np.array([0. for _ in range(num_cls)])
            per_cls_u = np.array([0. for _ in range(num_cls)])
            per_cls_strength = np.array([0. for _ in range(num_cls)])
            per_cls_conf_metrics = (per_cls_prob, per_cls_u, per_cls_strength)

        return (ood_metrics, calib_metrics, per_cls_conf_metrics)

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit, shape (N, K, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction, area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)

            # Mask ood examples
            if hasattr(self, "ood_indices"):
                ood_masker = self.get_ood_masker(seg_map)
                seg_map = seg_map[ood_masker]
                pred = pred[ood_masker]

            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the labels has been converted when dataset initialized
                    # in `get_palette_for_custom_classes ` this `label_map`
                    # should be `dict()`, see
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # for more ditails
                    label_map=dict(),
                    reduce_zero_label=self.reduce_zero_label))

        return pre_eval_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = class_names.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                # Get random state before set seed, and restore
                # random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU', 'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

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
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
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
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            if ret_metric in ('aAcc', 'IoU', 'Acc', 'Fscore', 'Precison', 'Recall', 'Dice')  # percentage metrics
            else np.round(np.nanmean(ret_metric_value), 2)  # other metrics
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics.pop('aNll', None)
        ret_metrics.pop('aEce1', None)
        ret_metrics.pop('aEce2', None)
        ret_metrics.pop('aBrierScore', None)

        ood_metrics = tuple(f"{a}.{b}" for a in ("max_prob", "max_logit", "entropy") for b in ("auroc", "aupr", "fpr95"))
        # remove ood metrics ret_metrics_summary
        for k in ood_metrics:
            ret_metrics_summary.pop(k, None)

        ood_metrics_summary = OrderedDict({ret_metric: np.round(np.nanmean(ret_metric_value), 2)
                                          for ret_metric, ret_metric_value in ret_metrics.items() if ret_metric in ood_metrics})

        for ret_metric in ood_metrics:
            ret_metrics.pop(ret_metric, None)

        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            if ret_metric in ('Acc', 'IoU')
            else np.round(ret_metric_value, 2)
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
            if key in ('aAcc', 'aNll', 'aEce1', 'aEce2', 'aBrierScore'):
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])
        ood_table_data = PrettyTable()
        for key, val in ood_metrics_summary.items():
            ood_table_data.add_column(key, [val])
        if not('roadanomaly' in str(type(self)).lower()):
            print_log('per class results:', logger)
            print_log('\n' + class_table_data.get_string(), logger=logger)
            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)
        print_log('OOD:', logger)
        if len(ood_metrics_summary):
            print_log('\n' + ood_table_data.get_string(), logger=logger)
        else:
            print_log("No image w/ OOD objects or all images have just OOD objects", logger)

        for key, value in ret_metrics_summary.items():
            if key in ('aAcc', 'aNll', 'aEce1', 'aEce2', 'aBrierScore'):
                eval_results[key] = value
            else:
                eval_results['m' + key] = value
        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx]
                for idx, name in enumerate(class_names)
            })
        for key, value in ood_metrics_summary.items():
            eval_results[key] = value
        return eval_results
