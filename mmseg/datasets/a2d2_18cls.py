# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class A2D2Dataset18Classes(CustomDataset):
    """The A2D2 dataset following the merged 18 class 'trainids' label format.

    The dataset features 41,280 frames with semantic segmentations having 18
    classes. This dataset configuration merging of the original semantic
    classes into a subset of 18 classes corresponding to the official benchmark
    results presented in the A2D2 paper (ref: p.8 "4. Experiment: Semantic
    segmentation"). Note that the 'background' class corresponding to 'blurred
    area' is removed with the dataset authors' consent due to ambiguity and
    sparseness.

    The 18 class segmentation conversion is specified in the following file:
        tools/convert_datasets/a2d2.py

    Instance segmentations and some segmentation classes are merged to comply
    with the categorical 'trainids' label format.
        Ex: 'Car 1' and 'Car 2' --> 'Car'

    The color palette approximately follows the Cityscapes coloring.

    The following segmentation classes are ignored (i.e. trainIds 255):
    - Rain dirt: Ambiguous semantic.

    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '_18LabelTrainIds.png' for the 18 class A2D2 dataset.

    Ref: https://www.a2d2.audi/a2d2/en/dataset.html
    """

    CLASSES = ('Road', 'Sky', 'Nature', 'Poles', 'Cars', 'Lane lines',
               'Buildings', 'Irrelevant', 'Traffic Info', 'Curb stones',
               'Side walk', 'Trucks', 'Grid structure',
               'Obstacles / trash on road', 'Pedestrians', 'Ego car',
               'Small traffic participants', 'Parking area')

    PALETTE = [[128, 64, 128], [70, 130, 180], [107, 142, 35], [153, 153, 153],
               [0, 0, 142], [255, 255, 255], [70, 70, 70], [250, 170, 30],
               [220, 220, 0], [230, 150, 140], [244, 35, 232], [190, 153, 153],
               [102, 102, 156], [0, 0, 230], [220, 20, 60], [0, 0, 70],
               [119, 11, 32], [250, 170, 160]]

    def __init__(self, **kwargs):
        super(A2D2Dataset18Classes, self).__init__(
            img_suffix='.png', seg_map_suffix='_18LabelTrainIds.png', **kwargs)
        assert osp.exists(self.img_dir) is not None

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
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
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

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
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
