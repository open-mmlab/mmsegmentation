import glob
import os

import cv2
import SimpleITK as sitk
from PIL import Image

root_path = 'data/'
img_suffix = '.tif'
seg_map_suffix = '.png'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

src_img_train_dir = os.path.join(root_path, 'CRASS/data_train')
src_mask_train_dir = os.path.join(root_path, 'CRASS/mask_mhd')
src_img_test_dir = os.path.join(root_path, 'CRASS/data_test')

tgt_img_train_dir = os.path.join(root_path, 'images/train/')
tgt_mask_train_dir = os.path.join(root_path, 'masks/train/')
tgt_img_test_dir = os.path.join(root_path, 'images/test/')
os.system('mkdir -p ' + tgt_img_train_dir)
os.system('mkdir -p ' + tgt_mask_train_dir)
os.system('mkdir -p ' + tgt_img_test_dir)


def filter_suffix_recursive(src_dir, suffix):
    suffix = '.' + suffix if '.' not in suffix else suffix
    file_paths = glob(
        os.path.join(src_dir, '**', '*' + suffix), recursive=True)
    file_names = [_.split('/')[-1] for _ in file_paths]
    return sorted(file_paths), sorted(file_names)


def read_single_array_from_med(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).squeeze()


def convert_meds_into_pngs(src_dir,
                           tgt_dir,
                           suffix='.dcm',
                           norm_min=0,
                           norm_max=255,
                           convert='RGB'):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    src_paths, src_names = filter_suffix_recursive(src_dir, suffix=suffix)
    num = len(src_paths)
    for i, (src_name, src_path) in enumerate(zip(src_names, src_paths)):
        tgt_name = src_name.replace(suffix, '.png')
        tgt_path = os.path.join(tgt_dir, tgt_name)

        img = read_single_array_from_med(src_path)
        if norm_min is not None and norm_max is not None:
            img = cv2.normalize(img, None, norm_min, norm_max, cv2.NORM_MINMAX,
                                cv2.CV_8U)
        pil = Image.fromarray(img).convert(convert)
        pil.save(tgt_path)
        print(f'processed {i+1}/{num}.')


convert_meds_into_pngs(
    src_img_train_dir,
    tgt_img_train_dir,
    suffix='.mhd',
    norm_min=0,
    norm_max=255,
    convert='RGB')

convert_meds_into_pngs(
    src_img_test_dir,
    tgt_img_test_dir,
    suffix='.mhd',
    norm_min=0,
    norm_max=255,
    convert='RGB')

convert_meds_into_pngs(
    src_mask_train_dir,
    tgt_mask_train_dir,
    suffix='.mhd',
    norm_min=0,
    norm_max=1,
    convert='L')
