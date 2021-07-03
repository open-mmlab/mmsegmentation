import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class A2D2Dataset(CustomDataset):
    """A2D2 dataset following the Cityscapes 'trainids' label format.

    The dataset features 41,280 frames with semantic segmentation in 38
    categories. Each pixel in an image is given a label describing the type of
    object it represents, e.g. pedestrian, car, vegetation, etc.

    NOTE: Instance segmentations and some segmentation classes are collapsed to
          follow the categorical 'trainids' label format.
          Ex: 'Car 1' and 'Car 2' --> 'Car'

          The segmentation conversion is defined in the following file:
              tools/convert_datasets/a2d2.py

          The color palette follows the coloring specified by 'class_list.json'.

    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '_labelTrainIds.png' for A2D2 dataset.

    Ref: https://www.a2d2.audi/a2d2/en/dataset.html
    """

    CLASSES = ('car', 'bicycle', 'pedestrian', 'truck', 'small_vehicles',
               'traffic_signal', 'traffic_sign', 'utility_vehicle',
               'sidebars', 'speed_bumber', 'curbstone', 'solid_line',
               'irrelevant_signs', 'road_blocks', 'tractor',
               'non-drivable_street', 'zebra_crossing', 'obstacles_trash',
               'poles', 'rd_restricted_area', 'animals', 'grid_structure',
               'signal_corpus', 'drivable_cobblestone', 'electronic_traffic',
               'slow_drive_area', 'nature_object', 'parking_area',
               'sidewalk', 'painted_drive_instr', 'traffic_guide_obj',
               'dashed_line', 'rd_normal_street', 'sky', 'buildings'
                )

    PALETTE = [[255, 0, 0], [182, 89, 6], [204, 153, 255], [255, 128, 0],
               [0, 255, 0], [0, 128, 255], [0, 255, 255], [255, 255, 0],
               [233, 100, 0], [110, 110, 0], [128, 128, 0], [255, 193, 37],
               [64, 0, 64], [185, 122, 87], [0, 0, 100], [139, 99, 108],
               [210, 50, 115], [255, 0, 128], [255, 246, 143], [150, 0, 150],
               [204, 255, 153], [238, 162, 173], [33, 44, 177], [180, 50, 180],
               [255, 70, 185], [238, 233, 191], [147, 253, 194], [150, 150, 200],
               [180, 150, 200], [200, 125, 210], [159, 121, 238], [128, 0, 255],
               [255, 0, 255], [135, 206, 255], [241, 230, 255]]

    def __init__(self, **kwargs):
        super(A2D2Dataset, self).__init__(
            img_suffix='.png', seg_map_suffix='_labelTrainIds.png', **kwargs)
        assert osp.exists(self.img_dir) is not None
