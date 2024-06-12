# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class TSP6KDataset(BaseSegDataset):
    """TSP6K dataset."""
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'railing',
                 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                 'truck', 'bus', 'motorcycle', 'bicycle', 'indication line',
                 'lane line', 'crosswalk', 'pole', 'traffic light',
                 'traffic sign'),
        palette=[[[128, 64, 128], [244, 35, 232], [70, 70, 70], [80, 90, 40],
                  [180, 165, 180], [107, 142, 35], [152, 251, 152],
                  [70, 130, 180], [255, 0, 0], [255, 100, 0], [0, 0, 142],
                  [0, 0, 70], [0, 60, 100], [0, 0, 230], [119, 11, 32],
                  [250, 170, 160], [250, 200, 160], [250, 240, 180],
                  [153, 153, 153], [250, 170, 30], [220, 220, 0]]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_sem.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
