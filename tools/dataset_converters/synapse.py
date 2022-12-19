import argparse
import os.path as osp

import numpy as np
from mmengine.utils import mkdir_or_exist
from PIL import Image
import h5py


def open_npz_file(file_path):
    file = np.load(file_path)
    return file['image'] * 255, file['label']


def open_h5_file(file_path):
    file = h5py.File(file_path)
    return np.array(file['image']) * 255, np.array(file['label'])


def read_files_from_txt(txt_path):
    with open(txt_path, "r") as f:
        files = f.readlines()
    files = [file.strip() for file in files]
    return files


def split_3d_image(img):
    c, _, _ = img.shape
    res = []
    for i in range(c):
        res.append(img[i, :, :])
    return res


def pares_args():
    parser = argparse.ArgumentParser(
        description='Convert synapse dataset to mmsegmentation format')
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='synapse dataset path.')
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
        raise ValueError(
            'The dataset path does not exist. '
            'Please enter a correct dataset path.')
    if not osp.exists(osp.join(dataset_path, "train_npz")) \
            or not osp.exists(osp.join(dataset_path, "test_vol_h5")):
        raise FileNotFoundError(
            'The dataset structure is incorrect. '
            'Please check your dataset.')

    train_list = read_files_from_txt(
        osp.join(dataset_path, "lists_lists_Synapse_train.txt"))

    test_list = read_files_from_txt(
        osp.join(dataset_path, "lists_lists_Synapse_test_vol.txt"))

    train_files = [osp.join(dataset_path, 'train_npz', file)
                   for file in train_list]

    test_files = [osp.join(dataset_path, 'test_vol_h5', file)
                  for file in test_list]

    mkdir_or_exist(osp.join(save_path, 'img_dir/train'))
    mkdir_or_exist(osp.join(save_path, 'img_dir/test'))
    mkdir_or_exist(osp.join(save_path, 'ann_dir/train'))
    mkdir_or_exist(osp.join(save_path, 'ann_dir/test'))

    for i, file in enumerate(train_files):
        img, label = open_npz_file(file + ".npz")
        img = Image.fromarray(img).convert('RGB')
        label = Image.fromarray(label).convert('L')
        img.save(osp.join(
            save_path, "img_dir/train", train_list[i] + ".jpg"))
        label.save(osp.join(
            save_path, "ann_dir/train", train_list[i] + ".png"))

    for i, file in enumerate(test_files):
        img_3d, label_3d = open_h5_file(file + ".npy.h5")
        imgs = split_3d_image(img_3d)
        labels = split_3d_image(label_3d)

        assert len(imgs) == len(labels),\
            "The length of images should be same as labels"

        for j, (image, label) in enumerate(zip(imgs, labels)):
            image = Image.fromarray(image).convert('RGB')
            label = Image.fromarray(label).convert('L')
            image.save(
                osp.join(save_path, "img_dir/test", test_list[i] + "_slice{}".format(str(j).rjust(4, "0")) + ".jpg"))
            label.save(
                osp.join(save_path, "ann_dir/test", test_list[i] + "_slice{}".format(str(j).rjust(4, "0")) + ".png"))


if __name__ == '__main__':
    main()
