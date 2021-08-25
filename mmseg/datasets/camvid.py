# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CamvidDataset(CustomDataset):

    def __init__(self, **params):
        CLASSES = ('background', 'aeroplane', 'bicycle', 'bird',
                   'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'table')

        PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70],
                   [102, 102, 156], [190, 153, 153], [153, 153, 153],
                   [250, 170, 30], [220, 220, 0], [107, 142, 35],
                   [152, 251, 152], [70, 130, 180], [220, 20, 60]]
        super(CamvidDataset, self).__init__(
            img_suffix='png',
            seg_map_suffix='png',
            **kwargs)
        assert osp.exists(self.img_dir)
