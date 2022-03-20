# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import tempfile
import zipfile

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert LoveDA dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='LoveDA folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'loveDA')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    assert 'Train.zip' in os.listdir(dataset_path), \
        'Train.zip is not in {}'.format(dataset_path)
    assert 'Val.zip' in os.listdir(dataset_path), \
        'Val.zip is not in {}'.format(dataset_path)
    assert 'Test.zip' in os.listdir(dataset_path), \
        'Test.zip is not in {}'.format(dataset_path)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for dataset in ['Train', 'Val', 'Test']:
            zip_file = zipfile.ZipFile(
                os.path.join(dataset_path, dataset + '.zip'))
            zip_file.extractall(tmp_dir)
            data_type = dataset.lower()
            for location in ['Rural', 'Urban']:
                for image_type in ['images_png', 'masks_png']:
                    if image_type == 'images_png':
                        dst = osp.join(out_dir, 'img_dir', data_type)
                    else:
                        dst = osp.join(out_dir, 'ann_dir', data_type)
                    if dataset == 'Test' and image_type == 'masks_png':
                        continue
                    else:
                        src_dir = osp.join(tmp_dir, dataset, location,
                                           image_type)
                        src_lst = os.listdir(src_dir)
                        for file in src_lst:
                            shutil.move(osp.join(src_dir, file), dst)
        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
