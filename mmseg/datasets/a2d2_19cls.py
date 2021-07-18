import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class A2D2Dataset19Classes(CustomDataset):
    """A2D2 dataset following the Cityscapes 'trainids' label format.

    The dataset features 41,280 frames with semantic segmentation in 35
    categories. Each pixel in an image is given a label describing the type of
    object it represents, e.g. pedestrian, car, vegetation, etc.

    This dataset variation merges original A2D2 classes into a subset emulating
    the classes found in Cityscapes.

    NOTE: Instance segmentations and some segmentation classes are collapsed to
          follow the categorical 'trainids' label format.
          Ex: 'Car 1' and 'Car 2' --> 'Car'

          The segmentation conversion is defined in the following file:
              tools/convert_datasets/a2d2.py

          The color palette follows the coloring in 'class_list.json'.

          The following segmentation classes are ignored (i.e. trainIds 255):
          - Ego car:  A calibrated system should a priori know what input
                      region corresponds to the ego vehicle.
          - Blurred area: Ambiguous semantic.
          - Rain dirt: Ambiguous semantic.

    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '_labelTrainIds.png' for A2D2 dataset.

    Ref: https://www.a2d2.audi/a2d2/en/dataset.html
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

    def __init__(self, **kwargs):
        super(A2D2Dataset19Classes, self).__init__(
            img_suffix='.png', seg_map_suffix='_19LabelTrainIds.png', **kwargs)
        assert osp.exists(self.img_dir) is not None
