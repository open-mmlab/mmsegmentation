import glob
import os
import shutil

import numpy as np
from PIL import Image
from scipy.io import loadmat

root_path = 'data/'
img_suffix = '.png'
seg_map_suffix = '.mat'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

x_train = glob.glob(os.path.join('data/CoNSeP/Train/Images/*' + img_suffix))
x_test = glob.glob(os.path.join('data/CoNSeP/Test/Images/*' + img_suffix))

os.system('mkdir -p ' + root_path + 'images/train/')
os.system('mkdir -p ' + root_path + 'images/val/')
os.system('mkdir -p ' + root_path + 'masks/train/')
os.system('mkdir -p ' + root_path + 'masks/val/')
D2_255_convert_dict = {0: 0, 255: 1}


def convert_2d(img, convert_dict=D2_255_convert_dict):
    arr_2d = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for c, i in convert_dict.items():
        arr_2d[img == c] = i
    return arr_2d


part_dir_dict = {0: 'CoNSeP/Train/', 1: 'CoNSeP/Test/'}
save_dir_dict = {0: 'train/', 1: 'val/'}
for ith, part in enumerate([x_train, x_test]):
    part_dir = part_dir_dict[ith]
    for img in part:
        basename = os.path.basename(img)
        shutil.copy(
            img, root_path + 'images/' + save_dir_dict[ith] +
            basename.split('.')[0] + save_img_suffix)

        mask_path = root_path + part_dir + 'Labels/' + basename.split(
            '.')[0] + seg_map_suffix
        label_ = loadmat(mask_path)
        label = label_['inst_map']
        label_type = label_['inst_type']
        label_dict = {i + 1: int(val) for i, val in enumerate(label_type)}

        save_mask_path = root_path + 'masks/' + save_dir_dict[
            ith] + basename.split('.')[0] + save_seg_map_suffix

        res = convert_2d(label, convert_dict=label_dict)
        res = Image.fromarray(res.astype(np.uint8))
        res.save(save_mask_path)
