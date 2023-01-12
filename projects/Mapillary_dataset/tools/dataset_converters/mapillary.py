# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp

import mmcv
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='Mapillary folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def RGB2Mask(image_path: str, dataset_version: str) -> None:
    """Mapillary Vistas Dataset provide 8-bit with color-palette class-specific
    labels for semantic segmentation.

    However, semantic segmentation needs single channel mask labels

    This code is about converting mapillary RGB labels
    {traing,validation/v1.2,v2.0/labels} to mask labels
    {{traing,validation/v1.2,v2.0/labels_mask}

    Args:
        image_path: image absolute path (str)
        dataset_version: v1.2 or v2.0 to choose color_map (str).
    """
    image = mmcv.imread(image_path)
    h, w, c = image.shape

    color_map_v1_2 = np.array([[165, 42, 42], [0, 192, 0], [196, 196, 196],
                               [190, 153, 153], [180, 165, 180],
                               [90, 120, 150], [102, 102, 156], [128, 64, 255],
                               [140, 140, 200], [170, 170,
                                                 170], [250, 170, 160],
                               [96, 96, 96], [230, 150, 140], [128, 64, 128],
                               [110, 110, 110], [244, 35,
                                                 232], [150, 100, 100],
                               [70, 70, 70], [150, 120, 90], [220, 20, 60],
                               [255, 0, 0], [255, 0, 100], [255, 0, 200],
                               [200, 128, 128], [255, 255, 255], [64, 170, 64],
                               [230, 160, 50], [70, 130, 180], [190, 255, 255],
                               [152, 251, 152], [107, 142, 35], [0, 170, 30],
                               [255, 255, 128], [250, 0, 30], [100, 140, 180],
                               [220, 220, 220], [220, 128, 128], [222, 40, 40],
                               [100, 170, 30], [40, 40, 40], [33, 33, 33],
                               [100, 128, 160], [142, 0, 0], [70, 100, 150],
                               [210, 170, 100], [153, 153,
                                                 153], [128, 128, 128],
                               [0, 0, 80], [250, 170, 30], [192, 192, 192],
                               [220, 220, 0], [140, 140, 20], [119, 11, 32],
                               [150, 0, 255], [0, 60, 100], [0, 0, 142],
                               [0, 0, 90], [0, 0, 230], [0, 80, 100],
                               [128, 64, 64], [0, 0, 110], [0, 0, 70],
                               [0, 0, 192], [32, 32, 32], [120, 10, 10],
                               [0, 0, 0]])

    color_map_v2_0 = np.array([[165, 42, 42], [0, 192, 0], [250, 170, 31],
                               [250, 170, 32], [196, 196,
                                                196], [190, 153, 153],
                               [180, 165, 180], [90, 120, 150], [250, 170, 33],
                               [250, 170, 34], [128, 128, 128], [250, 170, 35],
                               [102, 102, 156], [128, 64,
                                                 255], [140, 140,
                                                        200], [170, 170, 170],
                               [250, 170, 36], [250, 170, 160], [250, 170, 37],
                               [96, 96, 96], [230, 150, 140], [128, 64, 128],
                               [110, 110, 110], [110, 110, 110],
                               [244, 35, 232], [128, 196,
                                                128], [150, 100, 100],
                               [70, 70, 70], [150, 150, 150], [150, 120, 90],
                               [220, 20, 60], [220, 20, 60], [255, 0, 0],
                               [255, 0, 100], [255, 0, 200], [255, 255, 255],
                               [255, 255, 255], [250, 170, 29], [250, 170, 28],
                               [250, 170, 26], [250, 170, 25], [250, 170, 24],
                               [250, 170, 22], [250, 170, 21], [250, 170, 20],
                               [255, 255, 255], [250, 170, 19], [250, 170, 18],
                               [250, 170, 12], [250, 170, 11], [255, 255, 255],
                               [255, 255, 255], [250, 170, 16], [250, 170, 15],
                               [250, 170, 15], [255, 255,
                                                255], [255, 255, 255],
                               [255, 255, 255], [255, 255, 255], [64, 170, 64],
                               [230, 160, 50], [70, 130, 180], [190, 255, 255],
                               [152, 251, 152], [107, 142, 35], [0, 170, 30],
                               [255, 255, 128], [250, 0, 30], [100, 140, 180],
                               [220, 128, 128], [222, 40, 40], [100, 170, 30],
                               [40, 40, 40], [33, 33, 33], [100, 128, 160],
                               [20, 20, 255], [142, 0, 0], [70, 100, 150],
                               [250, 171, 30], [250, 172, 30], [250, 173, 30],
                               [250, 174, 30], [250, 175, 30], [250, 176, 30],
                               [210, 170, 100], [153, 153,
                                                 153], [153, 153, 153],
                               [128, 128, 128], [0, 0, 80], [210, 60, 60],
                               [250, 170, 30], [250, 170, 30], [250, 170, 30],
                               [250, 170, 30], [250, 170, 30], [250, 170, 30],
                               [192, 192, 192], [192, 192,
                                                 192], [192, 192, 192],
                               [220, 220, 0], [220, 220, 0], [0, 0, 196],
                               [192, 192, 192], [220, 220, 0], [140, 140, 20],
                               [119, 11, 32], [150, 0, 255], [0, 60, 100],
                               [0, 0, 142], [0, 0, 90], [0, 0, 230],
                               [0, 80, 100], [128, 64, 64], [0, 0, 110],
                               [0, 0, 70], [0, 0, 142], [0, 0, 192],
                               [170, 170, 170], [32, 32, 32], [111, 74, 0],
                               [120, 10, 10], [81, 0, 81], [111, 111, 0],
                               [0, 0, 0]])

    if dataset_version == 'v1.2':
        color_map = color_map_v1_2
    elif dataset_version == 'v2.0':
        color_map = color_map_v2_0

    flatten_v = np.matmul(
        image.reshape(-1, c),
        np.array([2, 3, 4]).reshape(3, 1))
    out = np.zeros_like(flatten_v)
    for idx, class_color in enumerate(color_map):
        value_idx = np.matmul(class_color, np.array([2, 3, 4]).reshape(3, 1))
        out[flatten_v == value_idx] = idx
    image = out.reshape(h, w)

    mmcv.imwrite(
        image.astype(np.uint8), image_path.replace('labels', 'labels_mask'))


def main():

    dataset_path = args.dataset_path

    if args.out_dir is None:
        out_dir = dataset_path
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(osp.join(out_dir, 'training', 'v1.2', 'labels_mask'))
    mkdir_or_exist(osp.join(out_dir, 'validation', 'v1.2', 'labels_mask'))
    mkdir_or_exist(osp.join(out_dir, 'training', 'v2.0', 'labels_mask'))
    mkdir_or_exist(osp.join(out_dir, 'validation', 'v2.0', 'labels_mask'))

    src_training_v1_2_list = glob.glob(
        os.path.join(dataset_path, 'training', 'v1.2', 'labels', '*.png'))
    src_validation_v1_2_list = glob.glob(
        os.path.join(dataset_path, 'validation', 'v1.2', 'labels', '*.png'))

    src_training_v2_0_list = glob.glob(
        os.path.join(dataset_path, 'training', 'v2.0', 'labels', '*.png'))
    src_validation_v2_0_list = glob.glob(
        os.path.join(dataset_path, 'validation', 'v2.0', 'labels', '*.png'))

    src_list_v1_2 = [src_training_v1_2_list, src_validation_v1_2_list]
    src_list_v2_0 = [src_training_v2_0_list, src_validation_v2_0_list]

    prog_bar = ProgressBar(len(src_list_v1_2))
    for data_file in src_list_v1_2:
        for _, src_path in enumerate(data_file):
            RGB2Mask(src_path, dataset_version='v1.2')
            prog_bar.update()
    print('v1.2 RGB labels have converted to Mask!')

    prog_bar = ProgressBar(len(src_list_v2_0))
    for data_file in src_list_v2_0:
        for _, src_path in enumerate(data_file):
            RGB2Mask(src_path, dataset_version='v2.0')
            prog_bar.update()
    print('v2.0 RGB labels have converted to Mask!')

    print('Have convert Mapillary RGB labels to Mask labels!')


if __name__ == '__main__':
    args = parse_args()
    main()
