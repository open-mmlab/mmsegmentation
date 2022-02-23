# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
from mmcv.utils import print_log

from ..utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class iSAIDDataset(CustomDataset):
    """ iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
    In segmentation map annotation for iSAID dataset, which is included
    in 16 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """

    CLASSES = ('background', 'ship', 'store_tank', 'baseball_diamond',
               'tennis_court', 'basketball_court', 'Ground_Track_Field',
               'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
               'Harbor')

    PALETTE = [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
               [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
               [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
               [0, 127, 191], [0, 127, 255], [0, 100, 155]]

    def __init__(self, **kwargs):
        super(iSAIDDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            ignore_index=255,
            **kwargs)
        assert osp.exists(self.img_dir)

    def load_annotations(self,
                         img_dir,
                         img_suffix,
                         ann_dir,
                         seg_map_suffix=None,
                         split=None):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    name = line.strip()
                    img_info = dict(filename=name + img_suffix)
                    if ann_dir is not None:
                        ann_name = name + '_instance_color_RGB'
                        seg_map = ann_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_img = img
                    seg_map = seg_img.replace(
                        img_suffix, '_instance_color_RGB' + seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos
