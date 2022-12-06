# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class DarkZurichDataset(CityscapesDataset):
    """DarkZurichDataset dataset."""

    def __init__(self,
                 img_suffix='_rgb_anon.png',
                 seg_map_suffix='_gt_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
