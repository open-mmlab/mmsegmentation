import glob
import os

import numpy as np
from PIL import Image

root_path = 'data/'
img_suffix = '.bmp'
seg_map_suffix = '.png'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

src_img_train_dir = os.path.join(
    root_path, 'JSRT/segmentation02/segmentation/org_train/')
src_mask_train_dir = os.path.join(
    root_path, 'JSRT/segmentation02/segmentation/label_train/')
src_img_test_dir = os.path.join(root_path,
                                'JSRT/segmentation02/segmentation/org_test/')
src_mask_test_dir = os.path.join(
    root_path, 'JSRT/segmentation02/segmentation/label_test/')

tgt_img_train_dir = os.path.join(root_path, 'images/train/')
tgt_mask_train_dir = os.path.join(root_path, 'masks/train/')
tgt_img_test_dir = os.path.join(root_path, 'images/test/')
tgt_mask_test_dir = os.path.join(root_path, 'masks/test/')
os.system('mkdir -p ' + tgt_img_train_dir)
os.system('mkdir -p ' + tgt_mask_train_dir)
os.system('mkdir -p ' + tgt_img_test_dir)
os.system('mkdir -p ' + tgt_mask_test_dir)


def filter_suffix_recursive(src_dir, suffix):
    suffix = '.' + suffix if '.' not in suffix else suffix
    file_paths = glob.glob(
        os.path.join(src_dir, '**', '*' + suffix), recursive=True)
    file_names = [_.split('/')[-1] for _ in file_paths]
    return sorted(file_paths), sorted(file_names)


def convert_label(img, convert_dict):
    arr = np.zeros_like(img, dtype=np.uint8)
    for c, i in convert_dict.items():
        arr[img == c] = i
    return arr


for i, src_img_dir in enumerate((src_img_train_dir, src_img_test_dir)):
    img_paths, img_names = filter_suffix_recursive(
        src_img_dir, suffix=img_suffix)
    if i == 0:
        tgt_img_dir = tgt_img_train_dir
    else:
        tgt_img_dir = tgt_img_test_dir

    for path, name in zip(img_paths, img_names):
        img = Image.open(path).convert('RGB')
        tgt_img_name = name.replace(img_suffix, save_img_suffix)
        tgt_img_path = os.path.join(tgt_img_dir, tgt_img_name)
        img.save(tgt_img_path)

for i, src_mask_dir in enumerate((src_mask_train_dir, src_mask_test_dir)):
    mask_paths, mask_names = filter_suffix_recursive(
        src_mask_dir, suffix=seg_map_suffix)
    if i == 0:
        tgt_mask_dir = tgt_mask_train_dir
    else:
        tgt_mask_dir = tgt_mask_test_dir

    for path, name in zip(mask_paths, mask_names):
        mask = np.array(Image.open(path).convert('L'))
        mask = convert_label(mask, convert_dict={0: 0, 85: 1, 170: 2, 255: 3})
        mask = Image.fromarray(mask)
        tgt_mask_name = name.replace(seg_map_suffix, save_seg_map_suffix)
        tgt_mask_path = os.path.join(tgt_mask_dir, tgt_mask_name)
        mask.save(tgt_mask_path)
