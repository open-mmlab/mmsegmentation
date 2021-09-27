# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import tempfile
import zipfile

import mmcv

DUT_OMRON_LEN = 5168


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DUT-OMRON dataset to mmsegmentation format')
    parser.add_argument('image_path', help='the path of DUT-OMRON-image.zip')
    parser.add_argument(
        'mask_path', help='the path of DUT-OMRON-gt-pixelwise.zip.zip')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.out_dir is None:
        out_dir = osp.join('data', 'DUT-OMRON')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))

    print('Generating images...')
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        zip_image = zipfile.ZipFile(args.image_path)
        zip_image.extractall(tmp_dir)
        zip_mask = zipfile.ZipFile(args.mask_path)
        zip_mask.extractall(tmp_dir)

        image_dir = osp.join(tmp_dir, 'DUT-OMRON-image')
        mask_dir = osp.join(tmp_dir, 'pixelwiseGT-new-PNG')

        assert len(os.listdir(image_dir)) == DUT_OMRON_LEN \
               and len(os.listdir(mask_dir)) == \
               DUT_OMRON_LEN, 'len(DUT-OMRON) != {}'.format(DUT_OMRON_LEN)

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
