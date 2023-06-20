import glob
import os

import numpy as np
from PIL import Image

root_path = 'data/'
img_suffix = '.jpg'
seg_map_suffix = '.png'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'
tgt_img_train_dir = os.path.join(root_path, 'images/train/')
tgt_mask_train_dir = os.path.join(root_path, 'masks/train/')
tgt_img_test_dir = os.path.join(root_path, 'images/test/')
os.system('mkdir -p ' + tgt_img_train_dir)
os.system('mkdir -p ' + tgt_mask_train_dir)
os.system('mkdir -p ' + tgt_img_test_dir)


def filter_suffix_recursive(src_dir, suffix):
    # filter out file names and paths in source directory
    suffix = '.' + suffix if '.' not in suffix else suffix
    file_paths = glob.glob(
        os.path.join(src_dir, '**/*' + suffix), recursive=True)
    file_names = [_.split('/')[-1] for _ in file_paths]
    return sorted(file_paths), sorted(file_names)


def convert_label(img, convert_dict):
    arr = np.zeros_like(img, dtype=np.uint8)
    for c, i in convert_dict.items():
        arr[img == c] = i
    return arr


def convert_pics_into_pngs(src_dir, tgt_dir, suffix, convert='RGB'):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    src_paths, src_names = filter_suffix_recursive(src_dir, suffix=suffix)

    for i, (src_name, src_path) in enumerate(zip(src_names, src_paths)):
        tgt_name = src_name.replace(suffix, save_img_suffix)
        tgt_path = os.path.join(tgt_dir, tgt_name)
        num = len(src_paths)
        img = np.array(Image.open(src_path))
        if len(img.shape) == 2:
            pil = Image.fromarray(img).convert(convert)
        elif len(img.shape) == 3:
            pil = Image.fromarray(img)
        else:
            raise ValueError('Input image not 2D/3D: ', img.shape)

        pil.save(tgt_path)
        print(f'processed {i+1}/{num}.')


def convert_label_pics_into_pngs(src_dir,
                                 tgt_dir,
                                 suffix,
                                 convert_dict={
                                     0: 2,
                                     128: 1,
                                     255: 0
                                 }):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    src_paths, src_names = filter_suffix_recursive(src_dir, suffix=suffix)
    num = len(src_paths)
    for i, (src_name, src_path) in enumerate(zip(src_names, src_paths)):
        tgt_name = src_name.replace(suffix, save_seg_map_suffix)
        tgt_path = os.path.join(tgt_dir, tgt_name)

        img = np.array(Image.open(src_path))
        img = convert_label(img, convert_dict)
        Image.fromarray(img).save(tgt_path)
        print(f'processed {i+1}/{num}.')


if __name__ == '__main__':

    convert_pics_into_pngs(
        os.path.join(
            root_path,
            'task3_disc_cup_segmentation/training/fundus color images/'),
        tgt_img_train_dir,
        suffix=img_suffix)

    convert_pics_into_pngs(
        os.path.join(
            root_path,
            'task3_disc_cup_segmentation/testing/fundus color images/'),
        tgt_img_test_dir,
        suffix=img_suffix)

    convert_label_pics_into_pngs(
        os.path.join(root_path,
                     'task3_disc_cup_segmentation/training/Disc_Cup_Mask/'),
        tgt_mask_train_dir,
        suffix=seg_map_suffix,
        convert_dict={
            0: 2,
            128: 1,
            255: 0
        })
    # original: [0, 128, 255] for ['optic cup', 'optic disc', 'background']
    # converted: [0, 1, 2] for ['background', 'optic disc', 'optic cup']
