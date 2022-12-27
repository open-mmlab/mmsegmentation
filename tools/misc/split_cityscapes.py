# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import numpy as np
from mmengine.fileio import FileClient
from mmengine.utils import mkdir_or_exist, track_parallel_progress


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type=str,
        help='The data root of cityscapes dataset.',
        default='./data/cityscapes/')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='The output directory of cityscapes semi-supervised annotations.',
        default='./data/cityscapes/semi_anns/')
    parser.add_argument(
        '--labeled-percent',
        type=float,
        nargs='+',
        help='The percentage of labeled data in the training set.',
        default=[1, 2, 5, 10])
    parser.add_argument(
        '--fold',
        type=int,
        help='K-fold cross validation for semi-supervised segmentation.',
        default=5)
    args = parser.parse_args()
    return args


def split_cityscapes(data_root: str, out_dir: str, percent: float, fold: int):

    def save_anns(name, images):
        sub_anns = dict()
        sub_anns['images'] = images

        mkdir_or_exist(out_dir)
        with open(osp.join(out_dir, name), 'w') as f:
            for line in images:
                print(line, file=f)

    file_client = FileClient(backend='disk')
    img_suffix = '_leftImg8bit.png'
    img_dir = osp.join(data_root, 'leftImg8bit/train')

    img_list = [
        p.replace(img_suffix, '') for p in file_client.list_dir_or_file(
            img_dir, list_dir=False, suffix=img_suffix, recursive=True)
    ]
    img_list.sort()
    np.random.seed(fold)
    labeled_total = int(percent / 100. * len(img_list))
    labeled_inds = set(
        np.random.choice(range(len(img_list)), size=labeled_total))
    labeled_images, unlabeled_images = [], []

    for i in range(len(img_list)):
        if i in labeled_inds:
            labeled_images.append(img_list[i])
        else:
            unlabeled_images.append(img_list[i])

    # save labeled and unlabeled
    labeled_name = f'cityscapes.{fold}@{percent}'
    unlabeled_name = f'cityscapes.{fold}@{percent}-unlabeled'

    save_anns(labeled_name, labeled_images)
    save_anns(unlabeled_name, unlabeled_images)


def multi_wrapper(args):
    return split_cityscapes(*args)


if __name__ == '__main__':
    args = parse_args()
    arguments_list = [(args.data_root, args.out_dir, p, f)
                      for f in range(1, args.fold + 1)
                      for p in args.labeled_percent]
    track_parallel_progress(multi_wrapper, arguments_list, args.fold)
