import os

import h5py
import numpy as np
from PIL import Image

root_path = 'data/'

tgt_img_train_dir = os.path.join(root_path, 'images/train/')
tgt_mask_train_dir = os.path.join(root_path, 'masks/train/')
tgt_img_val_dir = os.path.join(root_path, 'images/val/')
tgt_img_test_dir = os.path.join(root_path, 'images/test/')

os.system('mkdir -p ' + tgt_img_train_dir)
os.system('mkdir -p ' + tgt_mask_train_dir)
os.system('mkdir -p ' + tgt_img_val_dir)
os.system('mkdir -p ' + tgt_img_test_dir)


def extract_pics_from_h5(h5_path, h5_key, save_dir):
    f = h5py.File(h5_path, 'r')
    for i, img in enumerate(f[h5_key]):
        img = img.astype(np.uint8).squeeze()
        img = Image.fromarray(img)
        save_image_path = os.path.join(save_dir, str(i).zfill(8) + '.png')
        img.save(save_image_path)


if __name__ == '__main__':

    extract_pics_from_h5(
        'data/pcamv1/camelyonpatch_level_2_split_train_x.h5',
        h5_key='x',
        save_dir=tgt_img_train_dir)

    extract_pics_from_h5(
        'data/pcamv1/camelyonpatch_level_2_split_valid_x.h5',
        h5_key='x',
        save_dir=tgt_img_val_dir)

    extract_pics_from_h5(
        'data/pcamv1/camelyonpatch_level_2_split_test_x.h5',
        h5_key='x',
        save_dir=tgt_img_test_dir)

    extract_pics_from_h5(
        'data/pcamv1/camelyonpatch_level_2_split_train_mask.h5',
        h5_key='mask',
        save_dir=tgt_mask_train_dir)
