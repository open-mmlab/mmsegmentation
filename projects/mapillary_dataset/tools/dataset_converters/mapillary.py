# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import numpy as np
from mmengine.utils import (mkdir_or_exist, scandir, track_parallel_progress,
                            track_progress)
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary dataset RGB labels to mask labels')
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


def RGB2Mask(rgb_label_path: str) -> None:
    """Mapillary Vistas Dataset provide 8-bit with color-palette class-specific
    labels for semantic segmentation. However, semantic segmentation needs
    single channel mask labels.

    # Image.open().convert('P') -> img(h,w)     mask label
    # Image.open().convert('RGB') -> img(h,w,3) RGB label

    This code is about converting mapillary RGB labels
    {traing,validation/v1.2,v2.0/labels} to mask labels
    {{traing,validation/v1.2,v2.0/labels_mask}


    Args:
        rgb_label_path (str): image absolute path.
        dataset_version (str): v1.2 or v2.0 to choose color_map .
    """

    rgb_label = np.array(Image.open(rgb_label_path).convert('P'))

    masks_label = rgb_label

    mmcv.imwrite(
        masks_label.astype(np.uint8),
        rgb_label_path.replace('labels', 'labels_mask'))


def main():

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
        if 'labels' in label_path and 'labels_mask' not in label_path:
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
                RGB2Mask, RGB_labels_v1_2_path, nproc=args.nproc)
            print('Converting v2.0 ....')
            track_parallel_progress(
                RGB2Mask, RGB_labels_v2_0_path, nproc=args.nproc)
        elif args.version == 'v1.2':
            print('Converting v1.2 ....')
            track_parallel_progress(
                RGB2Mask, RGB_labels_v1_2_path, nproc=args.nproc)
        elif args.version == 'v2.0':
            print('Converting v2.0 ....')
            track_parallel_progress(
                RGB2Mask, RGB_labels_v2_0_path, nproc=args.nproc)

    else:
        if args.version == 'all':
            print('Converting v1.2 ....')
            track_progress(RGB2Mask, RGB_labels_v1_2_path)
            print('Converting v2.0 ....')
            track_progress(RGB2Mask, RGB_labels_v2_0_path)
        elif args.version == 'v1.2':
            print('Converting v1.2 ....')
            track_progress(RGB2Mask, RGB_labels_v1_2_path)
        elif args.version == 'v2.0':
            print('Converting v2.0 ....')
            track_progress(RGB2Mask, RGB_labels_v2_0_path)

    print('Have convert Mapillary Vistas Datasets RGB labels to Mask labels!')


if __name__ == '__main__':
    args = parse_args()
    main()
