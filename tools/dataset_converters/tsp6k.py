# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import shutil

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert TSP6K dataset to mmsegmentation format')
    parser.add_argument('raw_data', help='the path of raw data')
    parser.add_argument(
        '-o', '--out_dir', help='output path', default='./data/TSP6K')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mkdir_or_exist(args.out_dir)
    for subdir in ['label/val', 'image/val', 'label/train', 'image/train']:
        mkdir_or_exist(osp.join(args.out_dir, subdir))

    splits = ['train', 'val']

    for split in splits:
        split_path = osp.join(args.raw_data, f'split/{split}.txt')
        with open(split_path) as split_read:
            image_names = [line[:-1] for line in split_read.readlines()]
        for image_name in image_names:
            ori_label_path = osp.join(args.raw_data,
                                      f'label_trainval/{image_name}_sem.png')
            tar_label_path = osp.join(args.out_dir,
                                      f'label/{split}/{image_name}_sem.png')
            shutil.copy(ori_label_path, tar_label_path)

            ori_img_path = osp.join(args.raw_data, f'image/{image_name}.jpg')
            tar_img_path = osp.join(args.out_dir,
                                    f'image/{split}/{image_name}.jpg')
            shutil.copy(ori_img_path, tar_img_path)

    print('Done!')


if __name__ == '__main__':
    main()
