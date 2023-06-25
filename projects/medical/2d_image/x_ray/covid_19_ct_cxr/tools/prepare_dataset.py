import os

import numpy as np
from PIL import Image

root_path = 'data/'
src_img_dir = os.path.join(root_path, 'covid-chestxray-dataset', 'images')
src_mask_dir = os.path.join(root_path, 'covid-chestxray-dataset',
                            'annotations/lungVAE-masks')
tgt_img_train_dir = os.path.join(root_path, 'images/train/')
tgt_mask_train_dir = os.path.join(root_path, 'masks/train/')
tgt_img_test_dir = os.path.join(root_path, 'images/test/')
os.system('mkdir -p ' + tgt_img_train_dir)
os.system('mkdir -p ' + tgt_mask_train_dir)
os.system('mkdir -p ' + tgt_img_test_dir)


def convert_label(img, convert_dict):
    arr = np.zeros_like(img, dtype=np.uint8)
    for c, i in convert_dict.items():
        arr[img == c] = i
    return arr


if __name__ == '__main__':

    all_img_names = os.listdir(src_img_dir)
    all_mask_names = os.listdir(src_mask_dir)

    for img_name in all_img_names:
        base_name = img_name.replace('.png', '')
        base_name = base_name.replace('.jpg', '')
        base_name = base_name.replace('.jpeg', '')
        mask_name_orig = base_name + '_mask.png'
        if mask_name_orig in all_mask_names:
            mask_name = base_name + '.png'
            src_img_path = os.path.join(src_img_dir, img_name)
            src_mask_path = os.path.join(src_mask_dir, mask_name_orig)
            tgt_img_path = os.path.join(tgt_img_train_dir, img_name)
            tgt_mask_path = os.path.join(tgt_mask_train_dir, mask_name)

            img = Image.open(src_img_path).convert('RGB')
            img.save(tgt_img_path)
            mask = np.array(Image.open(src_mask_path))
            mask = convert_label(mask, {0: 0, 255: 1})
            mask = Image.fromarray(mask)
            mask.save(tgt_mask_path)
        else:
            src_img_path = os.path.join(src_img_dir, img_name)
            tgt_img_path = os.path.join(tgt_img_test_dir, img_name)
            img = Image.open(src_img_path).convert('RGB')
            img.save(tgt_img_path)
