import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class A2D2Dataset19Classes(CustomDataset):
    """The A2D2 dataset following the merged 19 class 'trainids' label format.

    The dataset features 41,280 frames with semantic segmentations having 19
    classes. This dataset configuration merging of the original semantic
    classes into a subset of 19 classes corresponding to the official benchmark
    results presented in the A2D2 paper (ref: p.8 "4. Experiment: Semantic
    segmentation").

    The 19 class segmentation conversion is specified in the following file:
        tools/convert_datasets/a2d2.py

    Instance segmentations and some segmentation classes are merged to comply
    with the categorical 'trainids' label format.
        Ex: 'Car 1' and 'Car 2' --> 'Car'

    The color palette approximately follows the Cityscapes coloring.

    The following segmentation classes are ignored (i.e. trainIds 255):
    - Rain dirt: Ambiguous semantic.

    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '_19LabelTrainIds.png' for the 19 class A2D2 dataset.

    Application of Cityscapes-like color palette:
        RGB              Cityscapes         A2D2 (19 classes)
        [128, 64, 128]   road           <-- Road
        [244, 35, 232]   sidewalk       <-- Side walk
        [70, 70, 70]     building       <-- Buildings
        [102, 102, 156]  wall           <-- Curb stones
        [190, 153, 153]  fence          <-- Irrelevant
        [153, 153, 153]  pole           <-- Poles
        [250, 170, 30]   traffic light  <-- Lane lines
        [220, 220, 0]    traffic sign   <-- Traffic Info
        [107, 142, 35]   vegetation     <-- Nature
        [152, 251, 152]  terrain        <-- Parking area
        [70, 130, 180]   sky            <-- Sky
        [220, 20, 60]    person         <-- Pedestrians
        [255, 0, 0]      rider          <-- Grid structure
        [0, 0, 142]      car            <-- Cars
        [0, 0, 70]       truck          <-- Trucks
        [0, 60, 100]     bus            <-- Background
        [0, 80, 100]     train          <-- Obstacles / trash on road
        [0, 0, 230]      motorcycle     <-- Ego car
        [119, 11, 32]    bicycle        <-- Small traffic participants

    Ref: https://www.a2d2.audi/a2d2/en/dataset.html
    """

    CLASSES = ('road', 'sidewalk', 'building', 'curbstone', 'irrelevant',
               'pole', 'lane line', 'traffic info', 'nature', 'parking area',
               'sky', 'pedestrian', 'grid structure', 'car', 'truck',
               'background', 'obstacles / trash on road', 'ego car',
               'small traffic participants')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, **kwargs):
        super(A2D2Dataset19Classes, self).__init__(
            img_suffix='.png', seg_map_suffix='_19LabelTrainIds.png', **kwargs)
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
