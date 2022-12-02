import argparse
import os
import os.path as osp
import random

import numpy as np
from mmengine.utils import mkdir_or_exist
from PIL import Image
import nibabel as nib


def open_nii_file(file_path):
    file = nib.load(file_path).get_fdata()
    return file


def split_image(image, label, direction, p=1, drop_background=True):
    imgs = []
    labels = []

    assert image.shape == label.shape, "image shape and label shape should be same"

    if direction == 0:
        for i in range(image.shape[0]):
            img = image[i, :, :]
            mask = label[i, :, :]

            if drop_background:
                if not np.any(mask):
                    continue

            if random.random() < p:
                imgs.append(img)
                labels.append(mask)
    elif direction == 1:
        for i in range(image.shape[1]):
            img = image[:, i, :]
            mask = label[:, i, :]

            if drop_background:
                if not np.any(mask):
                    continue

            if random.random() < p:
                imgs.append(img)
                labels.append(mask)
    else:
        for i in range(image.shape[2]):
            img = image[:, :, i]
            mask = label[:, :, i]

            if drop_background:
                if not np.any(mask):
                    continue

            if random.random() < p:
                imgs.append(img)
                labels.append(mask)
    return imgs, labels


def read_files_from_txt(txt_path):
    with open(txt_path, "r") as f:
        files = f.readlines()
    files = [file.strip() for file in files]
    return files


def pares_args():
    parser = argparse.ArgumentParser(
        description='Convert synapse dataset to mmsegmentation format')
    parser.add_argument('--dataset-path', type=str, help='synapse dataset path.')
    parser.add_argument('--drop-background', action='store_true', default=True,
                        help='discard pictures with blank labels.')
    parser.add_argument('--direction', default=2, type=int, help='3D image slice direction.')
    parser.add_argument("--p", default=1, type=float, help='the probability of extracting images')
    parser.add_argument('--save-path', default='data/synapse', type=str, help='save path of the dataset.')
    args = parser.parse_args()
    return args


def main():
    args = pares_args()
    dataset_path = args.dataset_path
    save_path = args.save_path

    if not osp.exists(dataset_path):
        raise ValueError("The dataset path does not exist. Please enter a correct dataset path.")

    if not osp.exists(osp.join(dataset_path, "img")) or not osp.exists(osp.join(dataset_path, "label")):
        raise FileNotFoundError("The dataset structure is incorrect. Please check your dataset.")

    mkdir_or_exist(osp.join(save_path, "img_dir"))
    mkdir_or_exist(osp.join(save_path, "ann_dir"))

    if osp.exists(osp.join(dataset_path, "train.txt")) and osp.exists(osp.join(dataset_path, "val.txt")):
        train_imgs_id = read_files_from_txt(osp.join(dataset_path, "train.txt"))
        val_imgs_id = read_files_from_txt(osp.join(dataset_path, "val.txt"))

        train_imgs_id = [id_.replace("img", "").replace(".nii.gz", "") for id_ in train_imgs_id]
        val_imgs_id = [id_.replace("img", "").replace(".nii.gz", "") for id_ in val_imgs_id]

        mkdir_or_exist(osp.join(save_path, "img_dir/train"))
        mkdir_or_exist(osp.join(save_path, "img_dir/val"))
        mkdir_or_exist(osp.join(save_path, "ann_dir/train"))
        mkdir_or_exist(osp.join(save_path, "ann_dir/val"))

        for i, img_id in enumerate(train_imgs_id):
            if osp.exists(osp.join(dataset_path, "img", "img" + img_id + ".nii.gz")) and osp.exists(
                    osp.join(dataset_path, "label", "label" + img_id + ".nii.gz")):
                img_nii = open_nii_file(osp.join(dataset_path, "img", "img" + img_id + ".nii.gz"))
                label_nii = open_nii_file(osp.join(dataset_path, "label", "label" + img_id + ".nii.gz"))

                imgs, masks = split_image(img_nii, label_nii, args.direction, args.p, args.drop_background)

                for j, (img, mask) in enumerate(zip(imgs, masks)):
                    save_name = img_id + "_slice" + str(j)
                    img = Image.fromarray(img)
                    mask = Image.fromarray(mask)
                    img.convert('RGB').save(osp.join(save_path, "img_dir", "train", save_name + ".jpg"))
                    mask.convert('L').save(osp.join(save_path, "ann_dir", "train", save_name + ".png"))
            else:
                continue

        for i, img_id in enumerate(val_imgs_id):
            if osp.exists(osp.join(dataset_path, "img", "img" + img_id + ".nii.gz")) and osp.exists(
                    osp.join(dataset_path, "label", "label" + img_id + ".nii.gz")):
                img_nii = open_nii_file(osp.join(dataset_path, "img", "img" + img_id + ".nii.gz"))
                label_nii = open_nii_file(osp.join(dataset_path, "label", "label" + img_id + ".nii.gz"))

                imgs, masks = split_image(img_nii, label_nii, args.direction, args.p, args.drop_background)

                for j, (img, mask) in enumerate(zip(imgs, masks)):
                    save_name = img_id + "_slice" + str(j)
                    img = Image.fromarray(img)
                    mask = Image.fromarray(mask)
                    img.convert('RGB').save(osp.join(save_path, "img_dir", "val", save_name + ".jpg"))
                    mask.convert('L').save(osp.join(save_path, "ann_dir", "val", save_name + ".png"))
            else:
                continue
    else:
        train_imgs_id = os.listdir(osp.join(dataset_path, "img"))
        train_imgs_id = [id_.replace("img", "").replace(".nii.gz", "") for id_ in train_imgs_id]
        mkdir_or_exist(osp.join(save_path, "img_dir"))
        mkdir_or_exist(osp.join(save_path, "ann_dir"))

        for i, img_id in enumerate(train_imgs_id):
            if osp.exists(osp.join(dataset_path, "img", "img" + img_id + ".nii.gz")) and osp.exists(
                    osp.join(dataset_path, "label", "label" + img_id + ".nii.gz")):
                img_nii = open_nii_file(osp.join(dataset_path, "img", "img" + img_id + ".nii.gz"))
                label_nii = open_nii_file(osp.join(dataset_path, "label", "label" + img_id + ".nii.gz"))

                imgs, masks = split_image(img_nii, label_nii, args.direction, args.p, args.drop_background)

                for j, (img, mask) in enumerate(zip(imgs, masks)):
                    save_name = img_id + "_slice" + str(j)
                    img = Image.fromarray(img)
                    mask = Image.fromarray(mask)
                    img.convert('RGB').save(osp.join(save_path, "img_dir", save_name + ".jpg"))
                    mask.convert('L').save(osp.join(save_path, "ann_dir", save_name + ".jpg"))
            else:
                continue


if __name__ == '__main__':
    main()
