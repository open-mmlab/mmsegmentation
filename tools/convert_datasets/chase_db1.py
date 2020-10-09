import argparse
import os
import os.path as osp
import shutil
import zipfile

import cv2
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CHASE_DB1 dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='path of CHASEDB1.zip')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'CHASE_DB1')
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

    tmp_dir = osp.join(out_dir, 'tmp')
    mmcv.mkdir_or_exist(tmp_dir)

    print('Extracting CHASEDB1.zip...')
    zip_file = zipfile.ZipFile(dataset_path)
    zip_file.extractall(tmp_dir)

    print('Generating training dataset...')
    for img_name in sorted(os.listdir(tmp_dir))[:60]:
        img = cv2.imread(osp.join(tmp_dir, img_name))
        if osp.splitext(img_name)[1] == '.jpg':
            cv2.imwrite(osp.join(out_dir, 'images', 'training', img_name), img)
        else:
            cv2.imwrite(
                osp.join(out_dir, 'annotations', 'training',
                         osp.splitext(img_name)[0] + '.jpg'),
                img[:, :, 0] // 128)

    for img_name in sorted(os.listdir(tmp_dir))[60:]:
        img = cv2.imread(osp.join(tmp_dir, img_name))
        if osp.splitext(img_name)[1] == '.jpg':
            cv2.imwrite(
                osp.join(out_dir, 'images', 'validation', img_name), img)
        else:
            cv2.imwrite(
                osp.join(out_dir, 'annotations', 'validation',
                         osp.splitext(img_name)[0] + '.jpg'),
                img[:, :, 0] // 128)

    print('Removing the temporary files...')
    shutil.rmtree(tmp_dir)

    print('Done!')


if __name__ == '__main__':
    main()
