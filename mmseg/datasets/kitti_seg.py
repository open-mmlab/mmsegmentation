# Copyright (c) OpenMMLab. All rights reserved.
from . import CityscapesDataset
from .builder import DATASETS


@DATASETS.register_module()
class KittiDataset(CityscapesDataset):

    # Kitti Seg Dataset
    def __init__(self, **kwargs):
        super(KittiDataset, self).__init__(
            img_suffix='_10.png', seg_map_suffix='_10.png', **kwargs)
