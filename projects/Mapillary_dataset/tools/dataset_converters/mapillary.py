# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from functools import partial

import mmcv
import numpy as np
from mmengine.utils import (mkdir_or_exist, scandir, track_parallel_progress,
                            track_progress)

colormap_v1_2 = np.array([[165, 42, 42], [0, 192, 0], [196, 196, 196],
                          [190, 153, 153], [180, 165, 180], [90, 120, 150],
                          [102, 102, 156], [128, 64, 255], [140, 140, 200],
                          [170, 170, 170], [250, 170, 160], [96, 96, 96],
                          [230, 150, 140], [128, 64, 128], [110, 110, 110],
                          [244, 35, 232], [150, 100, 100], [70, 70, 70],
                          [150, 120, 90], [220, 20, 60], [255, 0, 0],
                          [255, 0, 100], [255, 0, 200], [200, 128, 128],
                          [255, 255, 255], [64, 170, 64], [230, 160, 50],
                          [70, 130, 180], [190, 255, 255], [152, 251, 152],
                          [107, 142, 35], [0, 170, 30], [255, 255, 128],
                          [250, 0, 30], [100, 140, 180], [220, 220, 220],
                          [220, 128, 128], [222, 40, 40], [100, 170, 30],
                          [40, 40, 40], [33, 33, 33], [100, 128, 160],
                          [142, 0, 0], [70, 100, 150], [210, 170, 100],
                          [153, 153, 153], [128, 128, 128], [0, 0, 80],
                          [250, 170, 30], [192, 192, 192], [220, 220, 0],
                          [140, 140, 20], [119, 11, 32], [150, 0, 255],
                          [0, 60, 100], [0, 0, 142], [0, 0, 90], [0, 0, 230],
                          [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70],
                          [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]])

colormap_v2_0 = np.array([[165, 42, 42], [0, 192, 0], [250, 170, 31],
                          [250, 170, 32], [196, 196, 196], [190, 153, 153],
                          [180, 165, 180], [90, 120, 150], [250, 170, 33],
                          [250, 170, 34], [128, 128, 128], [250, 170, 35],
                          [102, 102, 156], [128, 64, 255], [140, 140, 200],
                          [170, 170, 170], [250, 170, 36], [250, 170, 160],
                          [250, 170, 37], [96, 96, 96], [230, 150, 140],
                          [128, 64, 128], [110, 110, 110], [110, 110, 110],
                          [244, 35, 232], [128, 196, 128], [150, 100, 100],
                          [70, 70, 70], [150, 150, 150], [150, 120, 90],
                          [220, 20, 60], [220, 20, 60], [255, 0, 0],
                          [255, 0, 100], [255, 0, 200], [255, 255, 255],
                          [255, 255, 255], [250, 170, 29], [250, 170, 28],
                          [250, 170, 26], [250, 170, 25], [250, 170, 24],
                          [250, 170, 22], [250, 170, 21], [250, 170, 20],
                          [255, 255, 255], [250, 170, 19], [250, 170, 18],
                          [250, 170, 12], [250, 170, 11], [255, 255, 255],
                          [255, 255, 255], [250, 170, 16], [250, 170, 15],
                          [250, 170, 15], [255, 255, 255], [255, 255, 255],
                          [255, 255, 255], [255, 255, 255], [64, 170, 64],
                          [230, 160, 50], [70, 130, 180], [190, 255, 255],
                          [152, 251, 152], [107, 142, 35], [0, 170, 30],
                          [255, 255, 128], [250, 0, 30], [100, 140, 180],
                          [220, 128, 128], [222, 40, 40], [100, 170, 30],
                          [40, 40, 40], [33, 33, 33], [100, 128, 160],
                          [20, 20, 255], [142, 0, 0], [70, 100, 150],
                          [250, 171, 30], [250, 172, 30], [250, 173, 30],
                          [250, 174, 30], [250, 175, 30], [250, 176, 30],
                          [210, 170, 100], [153, 153, 153], [153, 153, 153],
                          [128, 128, 128], [0, 0, 80], [210, 60, 60],
                          [250, 170, 30], [250, 170, 30], [250, 170, 30],
                          [250, 170, 30], [250, 170, 30], [250, 170, 30],
                          [192, 192, 192], [192, 192, 192], [192, 192, 192],
                          [220, 220, 0], [220, 220, 0], [0, 0, 196],
                          [192, 192, 192], [220, 220, 0], [140, 140, 20],
                          [119, 11, 32], [150, 0, 255], [0, 60, 100],
                          [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100],
                          [128, 64, 64], [0, 0, 110], [0, 0, 70], [0, 0, 142],
                          [0, 0, 192], [170, 170, 170], [32, 32, 32],
                          [111, 74, 0], [120, 10, 10], [81, 0, 81],
                          [111, 111, 0], [0, 0, 0]])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='Mapillary folder path')
    parser.add_argument(
        '--version',
        default='all',
        help="Mapillary labels version, 'v1.2','v2.0','all'")
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def mapillary_colormap2label(colormap: np.ndarray) -> list:
    """Create a `list` shaped (256^3, 1), convert each color palette to a
    number, which can use to find the correct label value.

    For example labels 0--Bird--[165, 42, 42]
    (165*256 + 42) * 256 + 42 = 10824234 (This is list's index])
    `colormap2label[10824234] = 0`

    In converting, if a RGB pixel value is [165, 42, 42],
    through colormap2label[10824234]-->can quickly find
    this labels value is 0.
    Through matrix multiply to compute a img is very fast.

    Args:
        colormap (np.ndarray):  Mapillary Vistas Dataset palette

    Returns:
        list: values are mask labels,
              indexes are  palette's convert results.、
    """
    colormap2label = np.zeros(256**3, dtype=np.longlong)
    for i, colormap_ in enumerate(colormap):
        colormap2label[(colormap_[0] * 256 + colormap_[1]) * 256 +
                       colormap_[2]] = i
    return colormap2label


def mapillary_masklabel(rgb_label: np.ndarray,
                        colormap2label: list) -> np.ndarray:
    """Computing a img mask label through `colormap2label` get in
    `mapillary_colormap2label(COLORMAP: np.ndarray)`

    Args:
        rgb_label (np.array): a RGB labels img.
        colormap2label (list): get in mapillary_colormap2label(colormap)

    Returns:
        np.ndarray: mask labels array.
    """
    colormap_ = rgb_label.astype('uint32')
    idx = np.array((colormap_[:, :, 0] * 256 + colormap_[:, :, 1]) * 256 +
                   colormap_[:, :, 2]).astype('uint32')
    return colormap2label[idx]


def RGB2Mask(rgb_label_path: str, colormap2label: list) -> None:
    """Mapillary Vistas Dataset provide 8-bit with color-palette class-specific
    labels for semantic segmentation. However, semantic segmentation needs
    single channel mask labels.

    This code is about converting mapillary RGB labels
    {traing,validation/v1.2,v2.0/labels} to mask labels
    {{traing,validation/v1.2,v2.0/labels_mask}

    Args:
        rgb_label_path (str): image absolute path.
        dataset_version (str): v1.2 or v2.0 to choose color_map .
    """
    rgb_label = mmcv.imread(rgb_label_path, channel_order='rgb')

    masks_label = mapillary_masklabel(rgb_label, colormap2label)

    mmcv.imwrite(
        masks_label.astype(np.uint8),
        rgb_label_path.replace('labels', 'labels_mask'))


def main():
    colormap2label_v1_2 = mapillary_colormap2label(colormap_v1_2)
    colormap2label_v2_0 = mapillary_colormap2label(colormap_v2_0)

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = dataset_path
    else:
        out_dir = args.out_dir

    RGB_labels_path = []
    RGB_labels_v1_2_path = []
    RGB_labels_v2_0_path = []
    print('Scanning labels path....')
    for label_path in scandir(dataset_path, suffix='.png', recursive=True):
        if 'labels' in label_path:
            rgb_label_path = osp.join(dataset_path, label_path)
            RGB_labels_path.append(rgb_label_path)
            if 'v1.2' in label_path:
                RGB_labels_v1_2_path.append(rgb_label_path)
            elif 'v2.0' in label_path:
                RGB_labels_v2_0_path.append(rgb_label_path)

    if args.version == 'all':
        print(f'Totaly found {len(RGB_labels_path)} {args.version} RGB labels')
    elif args.version == 'v1.2':
        print(f'Found {len(RGB_labels_v1_2_path)} {args.version} RGB labels')
    elif args.version == 'v2.0':
        print(f'Found {len(RGB_labels_v2_0_path)} {args.version} RGB labels')
    print('Making directories...')
    mkdir_or_exist(osp.join(out_dir, 'training', 'v1.2', 'labels_mask'))
    mkdir_or_exist(osp.join(out_dir, 'validation', 'v1.2', 'labels_mask'))
    mkdir_or_exist(osp.join(out_dir, 'training', 'v2.0', 'labels_mask'))
    mkdir_or_exist(osp.join(out_dir, 'validation', 'v2.0', 'labels_mask'))
    print('Directories Have Made...')

    if args.nproc > 1:
        if args.version == 'all':
            print('Converting v1.2 ....')
            track_parallel_progress(
                partial(RGB2Mask, colormap2label=colormap2label_v1_2),
                RGB_labels_v1_2_path,
                nproc=args.nproc)
            print('Converting v2.0 ....')
            track_parallel_progress(
                partial(RGB2Mask, colormap2label=colormap2label_v2_0),
                RGB_labels_v2_0_path,
                nproc=args.nproc)
        elif args.version == 'v1.2':
            print('Converting v1.2 ....')
            track_parallel_progress(
                partial(RGB2Mask, colormap2label=colormap2label_v1_2),
                RGB_labels_v1_2_path,
                nproc=args.nproc)
        elif args.version == 'v2.0':
            print('Converting v2.0 ....')
            track_parallel_progress(
                partial(RGB2Mask, colormap2label=colormap2label_v2_0),
                RGB_labels_v2_0_path,
                nproc=args.nproc)

    else:
        if args.version == 'all':
            print('Converting v1.2 ....')
            track_progress(
                partial(RGB2Mask, colormap2label=colormap2label_v1_2),
                RGB_labels_v1_2_path)
            print('Converting v2.0 ....')
            track_progress(
                partial(RGB2Mask, colormap2label=colormap2label_v2_0),
                RGB_labels_v2_0_path)
        elif args.version == 'v1.2':
            print('Converting v1.2 ....')
            track_progress(
                partial(RGB2Mask, colormap2label=colormap2label_v1_2),
                RGB_labels_v1_2_path)
        elif args.version == 'v2.0':
            print('Converting v2.0 ....')
            track_progress(
                partial(RGB2Mask, colormap2label=colormap2label_v2_0),
                RGB_labels_v2_0_path)

    print('Have convert Mapillary Vistas Datasets RGB labels to Mask labels!')


if __name__ == '__main__':
    args = parse_args()
    main()
