# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from typing import Callable, Dict, List, Optional, Sequence, Union
import os.path as osp

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose


@DATASETS.register_module()
class AI4Arctic(BaseSegDataset):
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        {'SIC_classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
         'SOD_classes': ['0', '1', '2', '3', '4', '5'],
         'FLOE_classes': ['0', '1', '2', '3', '4', '5', '6'],
         'SIC_palette': [[0, 0, 127],
                         [0, 0, 175],
                         [0, 0, 224],
                         [0, 0, 255],
                         [0, 42, 255],
                         [0, 85, 255],
                         [0, 127, 255],
                         [0, 169, 255],
                         [0, 212, 255],
                         [20, 255, 226],
                         [54, 255, 191]],
            'SOD_palette': [[123, 255, 123],
                            [157, 255, 89],
                            [191, 255, 54],
                            [226, 255, 20],
                            [255, 229, 0],
                            [255, 190, 0]],
            'FLOE_palette': [[255, 151, 0],
                             [255, 111, 0],
                             [255, 72, 0],
                             [255, 33, 0],
                             [224, 0, 0],
                             [175, 0, 0],
                             [127, 0, 0]]})

    def __init__(self,
                 img_suffix='.nc',
                 seg_map_suffix='.nc',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        from icecream import ic
        ic(self.ann_file)
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
