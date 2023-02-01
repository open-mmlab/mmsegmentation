# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import nibabel as nib
import numpy as np
from mmengine.utils import mkdir_or_exist
from PIL import Image


def read_files_from_txt(txt_path):
    with open(txt_path) as f:
        files = f.readlines()
    files = [file.strip() for file in files]
    return files


def read_nii_file(nii_path):
    img = nib.load(nii_path).get_fdata()
    return img


def split_3d_image(img):
    c, _, _ = img.shape
    res = []
    for i in range(c):
        res.append(img[i, :, :])
    return res


def label_mapping(label):
    """Label mapping from TransUNet paper setting. It only has 9 classes, which
    are 'background', 'aorta', 'gallbladder', 'left_kidney', 'right_kidney',
    'liver', 'pancreas', 'spleen', 'stomach', respectively. Other foreground
    classes in original dataset are all set to background.

    More details could be found here: https://arxiv.org/abs/2102.04306
    """
    maped_label = np.zeros_like(label)
    maped_label[label == 8] = 1
    maped_label[label == 4] = 2
    maped_label[label == 3] = 3
    maped_label[label == 2] = 4
    maped_label[label == 6] = 5
    maped_label[label == 11] = 6
    maped_label[label == 1] = 7
    maped_label[label == 7] = 8
    return maped_label


def pares_args():
    parser = argparse.ArgumentParser(
        description='Convert synapse dataset to mmsegmentation format')
    parser.add_argument(
        '--dataset-path', type=str, help='synapse dataset path.')
    parser.add_argument(
        '--save-path',
        default='data/synapse',
        type=str,
        help='save path of the dataset.')
    args = parser.parse_args()
    return args


def main():
    args = pares_args()
    dataset_path = args.dataset_path
    save_path = args.save_path

    if not osp.exists(dataset_path):
        raise ValueError('The dataset path does not exist. '
                         'Please enter a correct dataset path.')
    if not osp.exists(osp.join(dataset_path, 'img')) \
            or not osp.exists(osp.join(dataset_path, 'label')):
        raise FileNotFoundError('The dataset structure is incorrect. '
                                'Please check your dataset.')

    train_id = read_files_from_txt(osp.join(dataset_path, 'train.txt'))
    train_id = [idx[3:7] for idx in train_id]

    test_id = read_files_from_txt(osp.join(dataset_path, 'val.txt'))
    test_id = [idx[3:7] for idx in test_id]

    mkdir_or_exist(osp.join(save_path, 'img_dir/train'))
    mkdir_or_exist(osp.join(save_path, 'img_dir/val'))
    mkdir_or_exist(osp.join(save_path, 'ann_dir/train'))
    mkdir_or_exist(osp.join(save_path, 'ann_dir/val'))

    # It follows data preparation pipeline from here:
    # https://github.com/Beckschen/TransUNet/tree/main/datasets
    for i, idx in enumerate(train_id):
        img_3d = read_nii_file(
            osp.join(dataset_path, 'img', 'img' + idx + '.nii.gz'))
        label_3d = read_nii_file(
            osp.join(dataset_path, 'label', 'label' + idx + '.nii.gz'))

        img_3d = np.clip(img_3d, -125, 275)
        img_3d = (img_3d + 125) / 400
        img_3d *= 255
        img_3d = np.transpose(img_3d, [2, 0, 1])
        img_3d = np.flip(img_3d, 2)

        label_3d = np.transpose(label_3d, [2, 0, 1])
        label_3d = np.flip(label_3d, 2)
        label_3d = label_mapping(label_3d)

        for c in range(img_3d.shape[0]):
            img = img_3d[c]
            label = label_3d[c]

            img = Image.fromarray(img).convert('RGB')
            label = Image.fromarray(label).convert('L')
            img.save(
                osp.join(
                    save_path, 'img_dir/train', 'case' + idx.zfill(4) +
                    '_slice' + str(c).zfill(3) + '.jpg'))
            label.save(
                osp.join(
                    save_path, 'ann_dir/train', 'case' + idx.zfill(4) +
                    '_slice' + str(c).zfill(3) + '.png'))

    for i, idx in enumerate(test_id):
        img_3d = read_nii_file(
            osp.join(dataset_path, 'img', 'img' + idx + '.nii.gz'))
        label_3d = read_nii_file(
            osp.join(dataset_path, 'label', 'label' + idx + '.nii.gz'))

        img_3d = np.clip(img_3d, -125, 275)
        img_3d = (img_3d + 125) / 400
        img_3d *= 255
        img_3d = np.transpose(img_3d, [2, 0, 1])
        img_3d = np.flip(img_3d, 2)

        label_3d = np.transpose(label_3d, [2, 0, 1])
        label_3d = np.flip(label_3d, 2)
        label_3d = label_mapping(label_3d)

        for c in range(img_3d.shape[0]):
            img = img_3d[c]
            label = label_3d[c]

            img = Image.fromarray(img).convert('RGB')
            label = Image.fromarray(label).convert('L')
            img.save(
                osp.join(
                    save_path, 'img_dir/val', 'case' + idx.zfill(4) +
                    '_slice' + str(c).zfill(3) + '.jpg'))
            label.save(
                osp.join(
                    save_path, 'ann_dir/val', 'case' + idx.zfill(4) +
                    '_slice' + str(c).zfill(3) + '.png'))


if __name__ == '__main__':
    main()
