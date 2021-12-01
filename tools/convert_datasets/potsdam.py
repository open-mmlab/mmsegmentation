# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import mmcv
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='potsdam folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size', type=int, help='potsdam clip size', default=512)
    parser.add_argument(
        '--stride_size', type=int, help='potsdam overlap size', default=256)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, clip_save_dir, to_label=False):
    image = imread(image_path)

    h, w, c = image.shape
    cs = args.clip_size
    ss = args.stride_size

    num_rows = math.ceil((h - cs) / ss) if math.ceil(
        (h - cs) / ss) * ss + cs >= h else math.ceil((h - cs) / ss) + 1
    num_cols = math.ceil((w - cs) / ss) if math.ceil(
        (w - cs) / ss) * ss + cs >= w else math.ceil((w - cs) / ss) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * cs
    ymin = y * cs

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + cs > w, w - xmin - cs, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + cs > h, h - ymin - cs, np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + cs, w),
        np.minimum(ymin + cs, h)
    ],
                     axis=1)

    if to_label:
        color_map = np.array([[0, 0, 0], [255, 255, 255], [0, 0, 255],
                              [0, 255, 255], [0, 255, 0], [255, 255, 0],
                              [255, 0, 0]])
        flatten_v = np.matmul(
            image.reshape(-1, c),
            np.array([2, 3, 4]).reshape(3, 1))
        out = np.zeros_like(flatten_v)
        for idx, class_color in enumerate(color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        idx_i, idx_j = osp.basename(image_path).split('_')[2:4]
        imsave(
            osp.join(
                clip_save_dir, '%s_%s_%d_%d_%d_%d.png' %
                (idx_i, idx_j, start_x, start_y, end_x, end_y)),
            clipped_image.astype(np.uint8),
            check_contrast=False)


def main():
    splits = {
        'train': [
            '2_10',
            '2_11',
            '2_12',
            '3_10',
            '3_11',
            '3_12',
            '4_10',
            '4_11',
            '4_12',
            '5_10',
            '5_11',
            '5_12',
            '6_10',
            '6_11',
            '6_12',
            '6_7',
            '6_8',
            '6_9',
            '7_10',
            '7_11',
            '7_12',
            '7_7',
            '7_8',
            '7_9',
        ],
        'val': [
            '5_15',
            '6_15',
            '6_13',
            '3_13',
            '4_14',
            '6_14',
            '5_14',
            '2_13',
            '4_15',
            '2_14',
            '5_13',
            '4_13',
            '3_14',
            '7_13',
        ],
    }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'potsdam')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    zipp_list = glob.glob(os.path.join(dataset_path, '*.zip'))
    print('Find the data', zipp_list)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for zipp in zipp_list:
            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)
            src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
            for src_path in tqdm(src_path_list):
                idx_i, idx_j = osp.basename(src_path).split('_')[2:4]
                data_type = 'train' if '%s_%s' % (
                    idx_i, idx_j) in splits['train'] else 'val'
                if 'label' in src_path.split('_'):
                    dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                    clip_big_image(src_path, dst_dir, to_label=True)
                else:
                    dst_dir = osp.join(out_dir, 'img_dir', data_type)
                    clip_big_image(src_path, dst_dir, to_label=False)

        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main()
