# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import shutil

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert KITTI annotations to TrainIds')
    parser.add_argument('kitti_path', help='kitti data path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--ratio', default=0.25, type=float, help='test ratio splits')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    kitti_path = args.kitti_path
    out_dir = args.out_dir if args.out_dir else kitti_path
    mmcv.mkdir_or_exist(out_dir)

    # create the dir structure as same as cityscapes
    splits = ['train', 'val']

    for split in splits:
        mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', split))
        mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', split))

    training_dir = osp.join(kitti_path, 'training')

    original_dataset = list(
        mmcv.scandir(
            osp.join(training_dir),
            recursive=True
        )
    )

    images_path = sorted(
        [osp.join(training_dir, path)
         for path in original_dataset
         if 'image_2/' in path]
    )

    annotations_path = sorted(
        [osp.join(training_dir, path)
         for path in original_dataset
         if 'semantic/' in path]
    )

    # split the dataset into train and val
    # and copy the images and annotations to the new dir

    split_ratio = int(len(images_path) * (1 - args.ratio))

    for image in images_path[:split_ratio]:
        shutil.copy(image, osp.join(out_dir, 'img_dir', 'train'))

    for image in images_path[split_ratio:]:
        shutil.copy(image, osp.join(out_dir, 'img_dir', 'val'))

    for ann in annotations_path[:split_ratio]:
        shutil.copy(ann, osp.join(out_dir, 'ann_dir', 'train'))

    for ann in annotations_path[split_ratio:]:
        shutil.copy(ann, osp.join(out_dir, 'ann_dir', 'val'))

    print('Converted KITTI annotations to general format...')


if __name__ == '__main__':
    main()
