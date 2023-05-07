# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp

import mmcv
import numpy as np
from mmengine.utils import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert levir-cd dataset to mmsegmentation format')
    parser.add_argument('--dataset_path', help='potsdam folder path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=256)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_folder = args.dataset_path
    png_files = glob.glob(
        os.path.join(input_folder, '**/*.png'), recursive=True)
    output_folder = args.out_dir
    prog_bar = ProgressBar(len(png_files))
    for png_file in png_files:
        new_path = os.path.join(
            output_folder,
            os.path.relpath(os.path.dirname(png_file), input_folder))
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        label = False
        if 'label' in png_file:
            label = True
        clip_big_image(png_file, new_path, args, label)
        prog_bar.update()


def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    image = mmcv.imread(image_path)

    h, w, c = image.shape
    clip_size = args.clip_size
    stride_size = args.stride_size

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size
    ymin = y * clip_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ],
                     axis=1)

    if to_label:
        image[image == 255] = 1
        image = image[:, :, 0]
    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y, start_x:end_x] \
            if to_label else image[start_y:end_y, start_x:end_x, :]
        idx = osp.basename(image_path).split('.')[0]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(clip_save_dir,
                     f'{idx}_{start_x}_{start_y}_{end_x}_{end_y}.png'))


if __name__ == '__main__':
    main()
