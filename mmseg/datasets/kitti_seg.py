from .builder import DATASETS
from .custom import CustomDataset


# Register MyDataset class into DATASETS
@DATASETS.register_module()
class KittiSegDataset(CustomDataset):
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    # The formats of image and segmentation map are both .png in this case
    def __init__(self, **kwargs):
        super(KittiSegDataset, self).__init__(
            img_suffix='_10.png',
            seg_map_suffix='_10.png',
            **kwargs)
