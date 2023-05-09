# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil

import cv2
import mmcv


def kitti_to_train_ids(input):
    src, gt_dir, new_gt_dir = input
    label_file = src.replace('.png',
                             '_labelTrainIds.png').replace(gt_dir, new_gt_dir)
    img = cv2.imread(src)
    dirname = os.path.dirname(label_file)
    os.makedirs(dirname, exist_ok=True)
    sem_seg = img[:, :, 2]
    cv2.imwrite(label_file, sem_seg)


def copy_file(input):
    src, dst = input
    if not osp.exists(dst):
        os.makedirs(osp.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert KITTI-STEP annotations to TrainIds')
    parser.add_argument('kitti_path', help='kitti step data path')
    parser.add_argument('--gt-dir', default='panoptic_maps', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    kitti_path = args.kitti_path
    out_dir = args.out_dir if args.out_dir else kitti_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(kitti_path, args.gt_dir)

    ann_files = []
    for poly in mmcv.scandir(gt_dir, '.png', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        ann_files.append([poly_file, args.gt_dir, args.gt_dir + '_openmmlab'])

    if args.nproc > 1:
        mmcv.track_parallel_progress(kitti_to_train_ids, ann_files, args.nproc)
    else:
        mmcv.track_progress(kitti_to_train_ids, ann_files)

    copy_files = []
    for f in mmcv.scandir(gt_dir, '.png', recursive=True):
        original_f = osp.join(gt_dir, f).replace(args.gt_dir + '/train',
                                                 'training/image_02')
        new_f = osp.join(gt_dir, f).replace(args.gt_dir,
                                            'training_openmmlab/image_02')
        original_f = original_f.replace(args.gt_dir + '/val',
                                        'training/image_02')
        new_f = new_f.replace(args.gt_dir, 'training_openmmlab/image_02')
        copy_files.append([original_f, new_f])

    if args.nproc > 1:
        mmcv.track_parallel_progress(copy_file, copy_files, args.nproc)
    else:
        mmcv.track_progress(copy_file, copy_files)


if __name__ == '__main__':
    main()
