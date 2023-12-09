# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ACDC dataset to mmsegmentation format')
    parser.add_argument('raw_data', help='the path of raw data')
    parser.add_argument(
        '-o', '--out_dir', help='output path', default='./data/acdc')
    parser.add_argument(
        '--split',
        choices=['fog', 'night', 'rain', 'snow', 'all'],
        default='night')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    try:
        shutil.rmtree(args.out_dir)
        print('Directory removed successfully')
    except OSError as e:
        print(f'Error: {e.strerror}')

    print('Making directories...')
    mkdir_or_exist(args.out_dir)
    for subdir in ['gt/test', 'rgb_anno/test', 'gt/train', 'rgb_anno/train']:
        mkdir_or_exist(osp.join(args.out_dir, subdir))

    print('Moving images and annotations...')

    if args.split == 'all':
        anno_str_test = '/val/'
        gt_str_test = '/val/'
        anno_str_train = '/train/'
        gt_str_train = '/train/'
    else:
        anno_str_test = f'rgb_anon/{args.split}/val/'
        gt_str_test = f'gt/{args.split}/val/'
        anno_str_train = f'rgb_anon/{args.split}/train/'
        gt_str_train = f'gt/{args.split}/train/'

    for dirpath, _, filenames in os.walk(args.raw_data):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)

            if anno_str_test in full_path:
                if full_path.endswith('_rgb_anon.png'):
                    new_path = os.path.join(args.out_dir, 'rgb_anno/test',
                                            filename)
                    shutil.copy(full_path, new_path)
            elif gt_str_test in full_path:
                if full_path.endswith('_gt_labelTrainIds.png'):
                    new_path = os.path.join(args.out_dir, 'gt/test', filename)
                    shutil.copy(full_path, new_path)
            if anno_str_train in full_path:
                if full_path.endswith('_rgb_anon.png'):
                    new_path = os.path.join(args.out_dir, 'rgb_anno/train',
                                            filename)
                    shutil.copy(full_path, new_path)
            elif gt_str_train in full_path:
                if full_path.endswith('_gt_labelTrainIds.png'):
                    new_path = os.path.join(args.out_dir, 'gt/train', filename)
                    shutil.copy(full_path, new_path)

    print('Done!')


if __name__ == '__main__':
    main()
