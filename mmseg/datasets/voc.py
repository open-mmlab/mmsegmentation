import os.path as osp

import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module
class VOCDataset(CustomDataset):

    CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')

    def __init__(self, split, **kwargs):
        super(VOCDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

    @staticmethod
    def convert_to_color(seg):
        color_mat = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0],
                              [128, 128, 0], [0, 0, 128], [128, 0, 128],
                              [0, 128, 128], [128, 128, 128], [64, 0, 0],
                              [192, 0, 0], [64, 128, 0], [192, 128, 0],
                              [64, 0, 128], [192, 0, 128], [64, 128, 128],
                              [192, 128, 128], [0, 64, 0], [128, 64, 0],
                              [0, 192, 0], [128, 192, 0], [0, 64, 128]])
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3))
        for id in range(len(color_mat)):
            color_seg[seg == id, :] = color_mat[id]
        return color_seg
