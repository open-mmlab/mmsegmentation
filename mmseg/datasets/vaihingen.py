# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .potsdam import PotsdamDataset


@DATASETS.register_module()
class VaihingenDataset(PotsdamDataset):
    """ISPRS Vaihingen dataset.

    In segmentation map annotation for Vaihingen dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
