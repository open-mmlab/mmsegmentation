# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CIHPDataset(CustomDataset):
    """CIHP dataset.

    In segmentation map annotation for CIHP, 0 stands for background, which is
    included in 20 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'hat', 'hair', 'glove', 'sunglasses',
               'upperclothes', 'dress', 'coat', 'socks', 'pants', 'torsoSkin',
               'scarf', 'skirt', 'face', 'leftArm', 'rightArm', 'leftLeg',
               'rightLeg', 'leftShoe', 'rightShoe')

    PALETTE = [[0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51],
               [255, 85, 0], [0, 0, 85], [0, 119, 221], [85, 85,
                                                         0], [0, 85, 85],
               [85, 51, 0], [52, 86, 128], [0, 128, 0], [0, 0, 255],
               [51, 170, 221], [0, 255, 255], [85, 255, 170], [170, 255, 85],
               [255, 255, 0], [255, 170, 0]]

    def __init__(self, **kwargs):
        super(CIHPDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
