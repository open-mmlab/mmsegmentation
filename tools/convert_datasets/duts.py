# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import tempfile
import zipfile

import mmcv

TRAIN_LEN = 10553
TEST_LEN = 5019


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DUTS dataset to mmsegmentation format')
    parser.add_argument('trainset_path', help='the path of DUTS-TR.zip')
    parser.add_argument('testset_path', help='the path of DUTS-TE.zip')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.out_dir is None:
        out_dir = osp.join('data', 'DUTS')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))

    print('Generating images...')
    # DUTS-TR
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        zip_file = zipfile.ZipFile(args.trainset_path)
        zip_file.extractall(tmp_dir)
        image_dir = osp.join(tmp_dir, 'DUTS-TR', 'DUTS-TR-Image')
        mask_dir = osp.join(tmp_dir, 'DUTS-TR', 'DUTS-TR-Mask')

        assert len(os.listdir(image_dir)) == TRAIN_LEN \
               and len(os.listdir(mask_dir)) == \
               TRAIN_LEN, 'len(train_set) != {}'.format(TRAIN_LEN)

        for filename in sorted(os.listdir(image_dir)):
            shutil.copy(
                osp.join(image_dir, filename),
                osp.join(out_dir, 'images', 'training',
                         osp.splitext(filename)[0] + '.png'))

        for filename in sorted(os.listdir(mask_dir)):
            img = mmcv.imread(osp.join(mask_dir, filename))
            mmcv.imwrite(
                img[:, :, 0] // 128,
                osp.join(out_dir, 'annotations', 'training',
                         osp.splitext(filename)[0] + '.png'))

    # DUTS-TE
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        zip_file = zipfile.ZipFile(args.testset_path)
        zip_file.extractall(tmp_dir)
        image_dir = osp.join(tmp_dir, 'DUTS-TE', 'DUTS-TE-Image')
        mask_dir = osp.join(tmp_dir, 'DUTS-TE', 'DUTS-TE-Mask')

        assert len(os.listdir(image_dir)) == TEST_LEN \
               and len(os.listdir(mask_dir)) == \
               TEST_LEN, 'len(test_set) != {}'.format(TEST_LEN)

        for filename in sorted(os.listdir(image_dir)):
            shutil.copy(
                osp.join(image_dir, filename),
                osp.join(out_dir, 'images', 'validation',
                         osp.splitext(filename)[0] + '.png'))

        for filename in sorted(os.listdir(mask_dir)):
            img = mmcv.imread(osp.join(mask_dir, filename))
            mmcv.imwrite(
                img[:, :, 0] // 128,
                osp.join(out_dir, 'annotations', 'validation',
                         osp.splitext(filename)[0] + '.png'))

    print('Done!')


if __name__ == '__main__':
    main()
