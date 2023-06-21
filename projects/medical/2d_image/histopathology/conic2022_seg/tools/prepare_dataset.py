import glob
import os
import shutil

import numpy as np
from PIL import Image

img_save_root = 'data/'
root_path = 'data/'
img_suffix = '.png'
seg_map_suffix = '.png'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

label_set = set()


def save_masks_from_npz(data, save_root, part='masks/'):
    global label_set
    num = data.shape[0]
    for i in range(num):
        # np_img = data[i, :, :, :]
        np_mask = data[i, :, :, 1]
        label_set = set.union(label_set, set(np.unique(np_mask)))
        img = Image.fromarray(np_mask)
        save_path = os.path.join(save_root, part, str(i) + save_seg_map_suffix)
        img.save(save_path)


def save_images_from_npz(data, save_root, part='images/'):
    num = data.shape[0]
    for i in range(num):
        np_img = data[i, :, :, :]
        img = Image.fromarray(np_img)
        save_path = os.path.join(save_root, part, str(i) + save_img_suffix)
        img.save(save_path)


images_npy = np.load('data/CoNIC_Challenge/images.npy')
labels_npy = np.load('data/CoNIC_Challenge/labels.npy')

os.system('mkdir -p ' + img_save_root + 'images_ori')
os.system('mkdir -p ' + img_save_root + 'labels')
save_images_from_npz(images_npy, img_save_root, 'images_ori')
save_masks_from_npz(labels_npy, img_save_root, 'labels')
print(label_set)

x_train = glob.glob(os.path.join('data/images_ori/*' + img_suffix))

os.system('mkdir -p ' + root_path + 'images/train/')
os.system('mkdir -p ' + root_path + 'masks/train/')

part_dir_dict = {0: 'train/', 1: 'val/'}
for ith, part in enumerate([x_train]):
    part_dir = part_dir_dict[ith]
    for img in part:
        basename = os.path.basename(img)
        shutil.copy(
            img, root_path + 'images/' + part_dir + basename.split('.')[0] +
            save_img_suffix)
        mask_path = root_path + 'labels/' + basename.split(
            '.')[0] + seg_map_suffix
        save_mask_path = root_path + 'masks/' + part_dir + basename.split(
            '.')[0] + save_seg_map_suffix
        shutil.copy(mask_path, save_mask_path)
