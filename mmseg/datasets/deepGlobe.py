#mmseg/datasets/deepGlobe.py
# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata

@DATASETS.register_module()
class DeepGlobeDataset(BaseSegDataset):
    """Deep Globe Dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_t.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('Urban', 'Agriculture', 'Range', 'Forest', 'Water', 'Barren',
                 'Unknown'),
        palette=[[0,255,255], [255,255,0], [255,0,255], [0,255,0],
                 [0,0,255], [255,255,255], [1,1,1]
                 ])
    
    class_dict={
                "1": "Urban",
                "2": "Agriculture",
                "3": "Range",
                "4": "Forest",
                "5": "Water",
                "6": "Barren",
                "7": "Unknown"
                }
    color_map = [
            [0,255,255], [255,255,0], [255,0,255], [0,255,0],
            [0,0,255], [255,255,255], [1,1,1]
            ]

    def __init__(self,
                 img_suffix='_sat.jpg',
                 seg_map_suffix='_mask.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
