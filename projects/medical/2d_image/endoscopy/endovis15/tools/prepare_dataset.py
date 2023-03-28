import glob
import os

import numpy as np
import tifffile
from PIL import Image

root_path = 'data/'
img_suffix = '.tif'
seg_map_suffix = '.tif'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'
tgt_img_dir = os.path.join(root_path, 'images/train/')
tgt_mask_dir = os.path.join(root_path, 'masks/train/')
os.system('mkdir -p ' + tgt_img_dir)
os.system('mkdir -p ' + tgt_mask_dir)


def filter_suffix_recursive(src_dir, suffix='.png'):
    # filter out file names and paths in source directory
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


def convert_pics_into_pngs(src_dir, tgt_dir, suffix=img_suffix, convert='RGB'):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    src_paths, src_names = filter_suffix_recursive(src_dir, suffix=suffix)
    for i, (src_name, src_path) in enumerate(zip(src_names, src_paths)):
        tgt_name = src_name.replace(suffix, save_img_suffix)
        tgt_path = os.path.join(tgt_dir, tgt_name)

        # Image.open can read single-channel tif image,
        # but throws an error when reading 3-channel 16-bit one.
        # cv2 also reads this wrong as 3-channel GRAYSCALE image,
        # while it is in fact RGB (even IMREAD_UNCHANGED doesn't help).
        # So here we have to import tifffile package.
        img = tifffile.imread(src_path)

        if len(img.shape) == 2:
            pil = Image.fromarray(img).convert(convert)
        elif len(img.shape) == 3:
            pil = Image.fromarray(img)
        else:
            raise ValueError('Input image not 2D/3D: ', img.shape)

        pil.save(tgt_path)


def convert_label_pics_into_pngs(src_dir,
                                 tgt_dir,
                                 suffix='.png',
                                 convert_dict={
                                     0: 0,
                                     255: 1
                                 }):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    src_paths, src_names = filter_suffix_recursive(src_dir, suffix=suffix)
    num = len(src_paths)
    for i, (src_name, src_path) in enumerate(zip(src_names, src_paths)):
        tgt_name = src_name.replace(suffix, save_seg_map_suffix)
        tgt_path = os.path.join(tgt_dir, tgt_name)

        img = tifffile.imread(src_path)
        img = convert_label(img, convert_dict)
        Image.fromarray(img).save(tgt_path)
        print(f'processed {i+1}/{num}.')


convert_pics_into_pngs(
    os.path.join(root_path, '/EndoVis15/CVC-ClinicDB/Original'),
    os.path.join(tgt_img_dir),
    suffix=img_suffix)

convert_label_pics_into_pngs(
    os.path.join(root_path, 'EndoVis15/CVC-ClinicDB/Ground Truth'),
    os.path.join(tgt_mask_dir),
    suffix=seg_map_suffix)
