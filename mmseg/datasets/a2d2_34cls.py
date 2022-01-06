import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class A2D2Dataset34Classes(CustomDataset):
    """The A2D2 dataset with the original semantic segmentation classes.

    The dataset features 41,280 frames with semantic segmentation having 34
    classes. The original set of 38 classes are reduced to 34 for reasons
    explained below.

    The segmentation conversion is defined in the following file:
        tools/convert_datasets/a2d2.py

    Instance segmentations and some segmentation classes are collapsed to
    follow the categorical 'trainids' label format.
        Ex: 'Car 1' and 'Car 2' --> 'Car'

    The color palette follows the coloring in 'class_list.json'.

    The following segmentation classes are ignored (i.e. trainIds 255):
    - Ego car:  A calibrated system should a priori know what input
                region corresponds to the ego vehicle.
    - Blurred area: Ambiguous semantic.
    - Rain dirt: Ambiguous semantic.

    The following segmentation class is merged due to rarity:
    - Speed bumper --> RD normal street (randomly parsing 50% of dataset
    results in only one sample containing the 'speed_bumper' semantic)

    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '_34LabelTrainIds.png' for the 34 class A2D2 dataset.

    Ref: https://www.a2d2.audi/a2d2/en/dataset.html
    """

    CLASSES = ('rd_normal_street', 'non-drivable_street', 'rd_restricted_area',
               'drivable_cobblestone', 'slow_drive_area', 'parking_area',
               'solid_line', 'dashed_line', 'zebra_crossing', 'grid_structure',
               'traffic_guide_obj', 'painted_drive_instr', 'sidewalk',
               'curbstone', 'buildings', 'sidebars', 'road_blocks', 'poles',
               'traffic_signal', 'traffic_sign', 'signal_corpus',
               'irrelevant_signs', 'electronic_traffic', 'nature_object',
               'sky', 'pedestrian', 'bicycle', 'car', 'utility_vehicle',
               'truck', 'tractor', 'small_vehicles', 'animals',
               'obstacles_trash')

    PALETTE = [[255, 0, 255], [139, 99, 108], [150, 0, 150], [180, 50, 180],
               [238, 233, 191], [150, 150, 200], [255, 193, 37], [128, 0, 255],
               [210, 50, 115], [238, 162, 173], [159, 121,
                                                 238], [200, 125, 210],
               [180, 150, 200], [128, 128, 0], [241, 230, 255], [233, 100, 0],
               [185, 122, 87], [255, 246, 143], [0, 128, 255], [30, 220, 220],
               [33, 44, 177], [64, 0, 64], [255, 70, 185], [147, 253, 194],
               [135, 206, 255], [204, 153, 255], [182, 89, 6], [255, 0, 0],
               [255, 255, 0], [255, 128, 0], [0, 0, 100], [0, 255, 0],
               [204, 255, 153], [255, 0, 128]]

    def __init__(self, **kwargs):
        super(A2D2Dataset34Classes, self).__init__(
            img_suffix='.png', seg_map_suffix='_34LabelTrainIds.png', **kwargs)
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
