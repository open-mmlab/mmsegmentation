# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset
from collections import OrderedDict


@DATASETS.register_module()
class CityscapesDataset(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs):
        super(CityscapesDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        self.num_classes = 19

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

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)

        return result_files

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 imgfile_prefix=None):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove('cityscapes')
        if len(metrics) > 0:
            eval_results.update(
                super(CityscapesDataset,
                      self).evaluate(results, metrics, logger))

        return eval_results

    def _evaluate_cityscapes(self, results, logger, imgfile_prefix):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file

        Returns:
            dict[str: float]: Cityscapes evaluation results.
        """
        try:
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install cityscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_dir = imgfile_prefix

        eval_results = dict()
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        CSEval.args.evalInstLevelScore = True
        CSEval.args.predictionPath = osp.abspath(result_dir)
        CSEval.args.evalPixelAccuracy = True
        CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        for seg_map in mmcv.scandir(
                self.ann_dir, 'gtFine_labelIds.png', recursive=True):
            seg_map_list.append(osp.join(self.ann_dir, seg_map))
            pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

        eval_results.update(
            CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

        return eval_results

    def get_class_count(self, path="."):

        class_count_pixel = OrderedDict({i: 0 for i in range(len(self.CLASSES))})
        class_count_pixel[255] = 0  # ignore background
        class_count_semantic = OrderedDict({i: 0 for i in range(len(self.CLASSES))})
        class_count_semantic[255] = 0  # ignore background
        for index in range(self.__len__()):
            seg_gt = self.get_gt_seg_map_by_idx(index)
            if self.reduce_zero_label:
                seg_gt[seg_gt == 0] = 255
                seg_gt = seg_gt - 1
                seg_gt[seg_gt == 254] = 255
            for i in class_count_pixel.keys():
                class_count_pixel[i] += int((seg_gt == i).sum())
                class_count_semantic[i] += int((np.unique(seg_gt) == i).sum())
        self.class_count_pixel = np.array([*class_count_pixel.values()])
        self.class_count_semantic = np.array([*class_count_semantic.values()])

        filename = f"class_count_cityscapes_pixel.npy"
        with open(osp.join(path, filename), "wb") as f:
            np.save(f, self.class_count_pixel)

        filename = f"class_count_cityscapes_semantic.npy"
        with open(osp.join(path, filename), "wb") as f:
            np.save(f, self.class_count_semantic)

    def get_bags(self, mul=10):
        if not hasattr(self, "class_count_pixel"):
            try:
                with open("class_count_cityscapes_pixel.npy", "rb") as f:
                    self.class_count_pixel = np.load(f)
            except FileNotFoundError:
                self.get_class_count()
        class_count = self.class_count_pixel[:-1]
        low = float(1 / mul)
        hi = float(mul)
        used = np.zeros(self.num_classes, dtype=bool)
        bag_masks = []
        ratios = []
        bag_index = 0
        label2bag = {}
        bag_label_maps = []
        bags_classes = []
        bag_class_counts = []
        for cls in range(self.num_classes):
            if used[cls]:
                continue
            ratio_ = class_count / class_count[cls]
            ratios.append(ratio_)
            bag_mask = np.logical_and((ratio_ >= low), (ratio_ <= hi))
            if np.logical_and(bag_mask, used).any():
                # check if conflicts with used
                for c in np.where(np.logical_and(bag_mask, used))[0]:
                    conflict_bag_idx = label2bag[c]
                    conflict_bag_mask = bag_masks[conflict_bag_idx]
                    if bag_mask.sum() > conflict_bag_mask.sum():
                        bag_mask[c] = False
                    else:
                        conflict_bag_mask[c] = False
                        bag_masks[conflict_bag_idx] = conflict_bag_mask
            used = np.logical_or(used, bag_mask)
            bag_masks.append(bag_mask)

            for c in np.where(bag_mask)[0]:
                label2bag[c] = bag_index

            bag_index += 1
        num_bags = len(bag_masks)

        for i in range(num_bags):
            label_map = []
            for c in range(self.num_classes):
                if bag_masks[i][c]:
                    label_map.append(c)
                else:
                    label_map.append(self.num_classes + i)
            bag_label_maps.append(label_map)

            bag_clas_count = class_count[bag_masks[i]]
            bag_clas_count = np.append(bag_clas_count, class_count[~bag_masks[i]].sum())
            bag_class_counts.append(bag_clas_count)

            oth_mask = np.zeros(num_bags, dtype=bool)
            oth_mask[i] = True
            bag_masks[i] = np.concatenate((bag_masks[i], oth_mask))
            bags_classes.append([*np.where(bag_masks[i])[0]])

        assert all([bag_class_count.sum() == class_count.sum() for bag_class_count in bag_class_counts])
        assert np.sum([bag_mask.sum() for bag_mask in bag_masks]) == self.num_classes + num_bags

        self.num_bags = num_bags
        self.label2bag = label2bag
        self.bag_label_maps = bag_label_maps
        self.bag_masks = bag_masks
        self.bag_class_counts = bag_class_counts
        self.bags_classes = bags_classes
