import argparse
import os
import os.path as osp
import shutil
import zipfile

import cv2
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert HRF dataset to mmsegmentation format')
    parser.add_argument('healthy_path', help='the path of healthy.zip')
    parser.add_argument(
        'healthy_manualsegm_path', help='the path of healthy_manualsegm.zip')
    parser.add_argument('glaucoma_path', help='the path of glaucoma.zip')
    parser.add_argument(
        'glaucoma_manualsegm_path', help='the path of glaucoma_manualsegm.zip')
    parser.add_argument(
        'diabetic_retinopathy_path',
        help='the path of diabetic_retinopathy.zip')
    parser.add_argument(
        'diabetic_retinopathy_manualsegm_path',
        help='the path of diabetic_retinopathy_manualsegm.zip')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images_path = [
        args.healthy_path, args.glaucoma_path, args.diabetic_retinopathy_path
    ]
    annotations_path = [
        args.healthy_manualsegm_path, args.glaucoma_manualsegm_path,
        args.diabetic_retinopathy_manualsegm_path
    ]
    if args.out_dir is None:
        out_dir = osp.join('data', 'HRF')
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
    for now_path in images_path:
        tmp_dir = osp.join(out_dir, 'tmp')
        mmcv.mkdir_or_exist(tmp_dir)
        zip_file = zipfile.ZipFile(now_path)
        zip_file.extractall(tmp_dir)
        for filename in sorted(os.listdir(tmp_dir))[:5]:
            img = cv2.imread(osp.join(tmp_dir, filename))
            cv2.imwrite(
                osp.join(out_dir, 'images', 'training',
                         osp.splitext(filename)[0] + '.jpg'), img)
        for filename in sorted(os.listdir(tmp_dir))[5:]:
            img = cv2.imread(osp.join(tmp_dir, filename))
            cv2.imwrite(
                osp.join(out_dir, 'images', 'validation',
                         osp.splitext(filename)[0] + '.jpg'), img)
        shutil.rmtree(tmp_dir)

    print('Generating annotations...')
    for now_path in annotations_path:
        tmp_dir = osp.join(out_dir, 'tmp')
        mmcv.mkdir_or_exist(tmp_dir)
        zip_file = zipfile.ZipFile(now_path)
        zip_file.extractall(tmp_dir)
        for filename in sorted(os.listdir(tmp_dir))[:5]:
            img = cv2.imread(osp.join(tmp_dir, filename))
            cv2.imwrite(
                osp.join(out_dir, 'annotations', 'training',
                         osp.splitext(filename)[0] + '.jpg'),
                img[:, :, 0] // 128)
        for filename in sorted(os.listdir(tmp_dir))[5:]:
            img = cv2.imread(osp.join(tmp_dir, filename))
            cv2.imwrite(
                osp.join(out_dir, 'annotations', 'validation',
                         osp.splitext(filename)[0] + '.jpg'),
                img[:, :, 0] // 128)
        shutil.rmtree(tmp_dir)

    print('Done!')


if __name__ == '__main__':
    main()
