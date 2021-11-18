# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .sod_custom import SODCustomDataset


@DATASETS.register_module()
class HKUISDataset(SODCustomDataset):
    """HKU-IS dataset.

    In saliency map annotation for HKU-IS, 0 stands for background.
    ``reduce_zero_label`` is fixed to False. The ``img_suffix`` is fixed to
    '.png' and ``seg_map_suffix`` is fixed to '.png'.
    """

    CLASSES = ('background', 'foreground')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(HKUISDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
