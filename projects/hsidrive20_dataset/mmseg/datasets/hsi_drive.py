# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets import BaseSegDataset

# from mmseg.registry import DATASETS

classes_exp = ('unlabelled', 'road', 'road marks', 'vegetation',
               'painted metal', 'sky', 'concrete', 'pedestrian', 'water',
               'unpainted metal', 'glass')
palette_exp = [[0, 0, 0], [77, 77, 77], [255, 255, 255], [0, 255, 0],
               [255, 0, 0], [0, 0, 255], [102, 51, 0], [255, 255, 0],
               [0, 207, 250], [255, 166, 0], [0, 204, 204]]


# @DATASETS.register_module()
class HSIDrive20Dataset(BaseSegDataset):
    METAINFO = dict(classes=classes_exp, palette=palette_exp)

    def __init__(self,
                 img_suffix='.npy',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
