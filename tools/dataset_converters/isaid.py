# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import shutil
import tempfile
import zipfile

import mmcv
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist
from PIL import Image

iSAID_palette = \
    {
        0: (0, 0, 0),
        1: (0, 0, 63),
        2: (0, 63, 63),
        3: (0, 63, 0),
        4: (0, 63, 127),
        5: (0, 63, 191),
        6: (0, 63, 255),
        7: (0, 127, 63),
        8: (0, 127, 127),
        9: (0, 0, 127),
        10: (0, 0, 191),
        11: (0, 0, 255),
        12: (0, 191, 127),
        13: (0, 127, 191),
        14: (0, 127, 255),
        15: (0, 100, 155)
    }

iSAID_invert_palette = {v: k for k, v in iSAID_palette.items()}


def iSAID_convert_from_color(arr_3d, palette=iSAID_invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def slide_crop_image(src_path, out_dir, mode, patch_H, patch_W, overlap):
    img = np.asarray(Image.open(src_path).convert('RGB'))

    img_H, img_W, _ = img.shape

    if img_H < patch_H and img_W > patch_W:

        img = mmcv.impad(img, shape=(patch_H, img_W), pad_val=0)

        img_H, img_W, _ = img.shape

    elif img_H > patch_H and img_W < patch_W:

        img = mmcv.impad(img, shape=(img_H, patch_W), pad_val=0)

        img_H, img_W, _ = img.shape

    elif img_H < patch_H and img_W < patch_W:

        img = mmcv.impad(img, shape=(patch_H, patch_W), pad_val=0)

        img_H, img_W, _ = img.shape

    for x in range(0, img_W, patch_W - overlap):
        for y in range(0, img_H, patch_H - overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str -= diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str -= diff_y
                y_end = img_H

            img_patch = img[y_str:y_end, x_str:x_end, :]
            img_patch = Image.fromarray(img_patch.astype(np.uint8))
            image = osp.basename(src_path).split('.')[0] + '_' + str(
                y_str) + '_' + str(y_end) + '_' + str(x_str) + '_' + str(
                    x_end) + '.png'
            # print(image)
            save_path_image = osp.join(out_dir, 'img_dir', mode, str(image))
            img_patch.save(save_path_image, format='BMP')


def slide_crop_label(src_path, out_dir, mode, patch_H, patch_W, overlap):
    label = mmcv.imread(src_path, channel_order='rgb')
    label = iSAID_convert_from_color(label)
    img_H, img_W = label.shape

    if img_H < patch_H and img_W > patch_W:

        label = mmcv.impad(label, shape=(patch_H, img_W), pad_val=255)

        img_H = patch_H

    elif img_H > patch_H and img_W < patch_W:

        label = mmcv.impad(label, shape=(img_H, patch_W), pad_val=255)

        img_W = patch_W

    elif img_H < patch_H and img_W < patch_W:

        label = mmcv.impad(label, shape=(patch_H, patch_W), pad_val=255)

        img_H = patch_H
        img_W = patch_W

    for x in range(0, img_W, patch_W - overlap):
        for y in range(0, img_H, patch_H - overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str -= diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str -= diff_y
                y_end = img_H

            lab_patch = label[y_str:y_end, x_str:x_end]
            lab_patch = Image.fromarray(lab_patch.astype(np.uint8), mode='P')

            image = osp.basename(src_path).split('.')[0].split(
                '_')[0] + '_' + str(y_str) + '_' + str(y_end) + '_' + str(
                    x_str) + '_' + str(x_end) + '_instance_color_RGB' + '.png'
            lab_patch.save(osp.join(out_dir, 'ann_dir', mode, str(image)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert iSAID dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='iSAID folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')

    parser.add_argument(
        '--patch_width',
        default=896,
        type=int,
        help='Width of the cropped image patch')
    parser.add_argument(
        '--patch_height',
        default=896,
        type=int,
        help='Height of the cropped image patch')
    parser.add_argument(
        '--overlap_area', default=384, type=int, help='Overlap area')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    # image patch width and height
    patch_H, patch_W = args.patch_width, args.patch_height

    overlap = args.overlap_area  # overlap area

    if args.out_dir is None:
        out_dir = osp.join('data', 'iSAID')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))

    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'test'))

    assert os.path.exists(os.path.join(dataset_path, 'train')), \
        f'train is not in {dataset_path}'
    assert os.path.exists(os.path.join(dataset_path, 'val')), \
        f'val is not in {dataset_path}'
    assert os.path.exists(os.path.join(dataset_path, 'test')), \
        f'test is not in {dataset_path}'

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for dataset_mode in ['train', 'val', 'test']:

            # for dataset_mode in [ 'test']:
            print(f'Extracting  {dataset_mode}ing.zip...')
            img_zipp_list = glob.glob(
                os.path.join(dataset_path, dataset_mode, 'images', '*.zip'))
            print('Find the data', img_zipp_list)
            for img_zipp in img_zipp_list:
                zip_file = zipfile.ZipFile(img_zipp)
                zip_file.extractall(os.path.join(tmp_dir, dataset_mode, 'img'))
            src_path_list = glob.glob(
                os.path.join(tmp_dir, dataset_mode, 'img', 'images', '*.png'))

            src_prog_bar = ProgressBar(len(src_path_list))
            for i, img_path in enumerate(src_path_list):
                if dataset_mode != 'test':
                    slide_crop_image(img_path, out_dir, dataset_mode, patch_H,
                                     patch_W, overlap)

                else:
                    shutil.move(img_path,
                                os.path.join(out_dir, 'img_dir', dataset_mode))
                src_prog_bar.update()

            if dataset_mode != 'test':
                label_zipp_list = glob.glob(
                    os.path.join(dataset_path, dataset_mode, 'Semantic_masks',
                                 '*.zip'))
                for label_zipp in label_zipp_list:
                    zip_file = zipfile.ZipFile(label_zipp)
                    zip_file.extractall(
                        os.path.join(tmp_dir, dataset_mode, 'lab'))

                lab_path_list = glob.glob(
                    os.path.join(tmp_dir, dataset_mode, 'lab', 'images',
                                 '*.png'))
                lab_prog_bar = ProgressBar(len(lab_path_list))
                for i, lab_path in enumerate(lab_path_list):
                    slide_crop_label(lab_path, out_dir, dataset_mode, patch_H,
                                     patch_W, overlap)
                    lab_prog_bar.update()

        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
