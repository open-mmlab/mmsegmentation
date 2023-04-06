# from scipy.io import loadmat
# import pandas as pd
# import random
# from sklearn.model_selection import train_test_split
# import os
# import shutil
# import glob
# # import mmcv
# import numpy as np
# from PIL import Image
# import cv2

# root_path = "/mnt/petrelfs/guosizheng/gsz/2d_fenge_todo/IDRID/"
# img_suffix='.jpg'
# seg_map_suffix='.tif'
# save_img_suffix='.png'
# save_seg_map_suffix='.png'

# all_imgs = glob.glob(os.path.join(root_path, "Segmentation/A. Segmentation/1. Original Images/a. Training Set/*" + img_suffix))  # noqa

# # x_train, x_test = train_test_split(all_imgs, test_size = 0.2, random_state=0)  # noqa
# x_train = all_imgs
# x_test = glob.glob(os.path.join(root_path, "Segmentation/A. Segmentation/1. Original Images/b. Testing Set/*" + img_suffix))  # noqa

# print(len(x_train), len(x_test))
# os.system("mkdir -p " + root_path +"images/train/")
# os.system("mkdir -p " + root_path +"images/test/")
# os.system("mkdir -p " + root_path +"masks/train/")
# os.system("mkdir -p " + root_path +"masks/test/")
# D2_255_convert_dict = {0:0, 255:1}
# def convert_2d(img, convert_dict=D2_255_convert_dict):
#     arr_2d = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
#     for c, i in convert_dict.items():
#         arr_2d[img == c] = i
#     return arr_2d
# type_part_dict = {0:"_MA", 1:"_HE", 2:"_EX", 3:"_SE", 4:"_OD"}
# part_dir_dict={0:"train/", 1:"test/"}
# for ith, part in enumerate([x_train, x_test]):
#     part_dir = part_dir_dict[ith]
#     for img in part:
#         basename = os.path.basename(img)
#         image = Image.open(img)
#         w, h = image.size[:2]
#         img_save_path = root_path +"images/" + part_dir + basename.split('.')[0] + save_img_suffix  # noqa
#         image.save(img_save_path)

#         masks = np.zeros((h, w, 5), dtype = np.uint8)
#         for type_i, type_part in enumerate(["1. Microaneurysms/", "2. Haemorrhages/", "3. Hard Exudates/", "4. Soft Exudates/", "5. Optic Disc/"]):  # noqa
#             instance_mask_path = os.path.join(root_path, "Segmentation/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/", type_part, basename.split('.')[0] + type_part_dict[type_i] + seg_map_suffix)  # noqa
#             if os.path.exists(instance_mask_path):
#                 masks[:, :, type_i] = np.array(Image.open(instance_mask_path)).astype(np.uint8)  # noqa
#                 # print(np.unique(masks[:, :, type_i]))
#                 masks[:, :, type_i][masks[:, :, type_i] == 1] =  type_i + 1

#         save_mask_path = root_path +"masks/" + part_dir + basename.split('.')[0] + save_seg_map_suffix  # noqa
#         masks = np.max(masks, axis = 2)
#         print(np.unique(masks))
#         masks = Image.fromarray(masks)
#         masks.save(save_mask_path)

import argparse
import glob
import os

from sklearn.model_selection import train_test_split


def save_anno(img_list, file_path, remove_suffix=True):
    if remove_suffix:  # （文件路径从data/${image/masks}之后的相对路径开始）
        img_list = [
            '/'.join(img_path.split('/')[-2:]) for img_path in img_list
        ]  # noqa
        img_list = [
            '.'.join(img_path.split('.')[:-1]) for img_path in img_list
        ]  # noqa
    with open(file_path, 'w') as file_:
        for x in list(img_list):
            file_.write(x + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', default='data/')
    args = parser.parse_args()
    data_root = args.data_root
    if os.path.exists(os.path.join(data_root, 'masks/val')):
        x_val = sorted(glob.glob(data_root + '/images/val/*.png'))
        save_anno(x_val, data_root + '/val.txt')
    if os.path.exists(os.path.join(data_root, 'masks/test')):
        x_test = sorted(glob.glob(data_root + '/images/test/*.png'))
        save_anno(x_test, data_root + '/test.txt')
    if not os.path.exists(os.path.join(
            data_root, 'masks/val')) and not os.path.exists(
                os.path.join(data_root, 'masks/test')):  # noqa
        all_imgs = sorted(glob.glob(data_root + '/images/train/*.png'))
        x_train, x_val = train_test_split(
            all_imgs, test_size=0.2, random_state=0)  # noqa
        save_anno(x_train, data_root + '/train.txt')
        save_anno(x_val, data_root + '/val.txt')
    else:
        x_train = sorted(glob.glob(data_root + '/images/train/*.png'))
        save_anno(x_train, data_root + '/train.txt')
    # ---------生成md5值以及包含无标签image的list，pr时该部分代码将被删除--------------
    import hashlib
    all_imgs = []
    for fpath, dirname, fnames in os.walk(os.path.join(data_root, 'images')):
        for fname in fnames:
            all_imgs.append(os.path.join(fpath, fname))
    f_ = open(data_root + '/images_md5_list.txt', 'w')
    for img in sorted(all_imgs):
        with open(img, 'rb') as fd:
            fmd5 = hashlib.md5(fd.read()).hexdigest()
        f_.write(fmd5 + '\t' + '/'.join(img.split('/')[-2:]) + '\n')
    f_.close()
