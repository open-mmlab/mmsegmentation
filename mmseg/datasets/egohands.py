# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class EgoHandsDataset(CustomDataset):
    """Egohands dataset.

    In segmentation map annotation for egohands, 0 stands for background,
    1, 2: person 1 left, right hand
    3, 4: person 2 left, right hand
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.jpg'.
    """

    CLASSES = ('background', 'hand1_lt', 'hand1_rt', 'hand2_lt', 'hand2_rt')

    PALETTE = [[120, 120, 120],
               [6, 230, 230], [255, 0, 0],
               [0, 0, 142], [0, 0, 70]]

    def __init__(self, **kwargs):
        super(EgoHandsDataset, self).__init__(
            img_suffix='',
            seg_map_suffix='',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
