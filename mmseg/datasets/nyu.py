# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class NYUDataset(BaseSegDataset):
    """NYU depth estimation dataset. The file structure should be.

    .. code-block:: none

        ├── data
        │   ├── nyu
        │   │   ├── images
        │   │   │   ├── train
        │   │   │   │   ├── scene_xxx.jpg
        │   │   │   │   ├── ...
        │   │   │   ├── test
        │   │   ├── annotations
        │   │   │   ├── train
        │   │   │   │   ├── scene_xxx.png
        │   │   │   │   ├── ...
        │   │   │   ├── test

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as
            specify classes to load. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img_path='images', depth_map_path='annotations').
        img_suffix (str): Suffix of images. Default: '.jpg'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=True``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default to False.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """
    METAINFO = dict(
        classes=('printer_room', 'bathroom', 'living_room', 'study',
                 'conference_room', 'study_room', 'kitchen', 'home_office',
                 'bedroom', 'dinette', 'playroom', 'indoor_balcony',
                 'laundry_room', 'basement', 'excercise_room', 'foyer',
                 'home_storage', 'cafe', 'furniture_store', 'office_kitchen',
                 'student_lounge', 'dining_room', 'reception_room',
                 'computer_lab', 'classroom', 'office', 'bookstore'))

    def __init__(self,
                 data_prefix=dict(
                     img_path='images', depth_map_path='annotations'),
                 img_suffix='.jpg',
                 depth_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            data_prefix=data_prefix,
            img_suffix=img_suffix,
            seg_map_suffix=depth_map_suffix,
            **kwargs)

    def _get_category_id_from_filename(self, image_fname: str) -> int:
        """Retrieve the category ID from the given image filename."""
        image_fname = osp.basename(image_fname)
        position = image_fname.find(next(filter(str.isdigit, image_fname)), 0)
        categoty_name = image_fname[:position - 1]
        if categoty_name not in self._metainfo['classes']:
            return -1
        else:
            return self._metainfo['classes'].index(categoty_name)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('depth_map_path', None)

        _suffix_len = len(self.img_suffix)
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                depth_map = img[:-_suffix_len] + self.seg_map_suffix
                data_info['depth_map_path'] = osp.join(ann_dir, depth_map)
            data_info['seg_fields'] = []
            data_info['category_id'] = self._get_category_id_from_filename(img)
            data_list.append(data_info)
        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
