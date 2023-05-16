import os

import numpy as np
from PIL import Image

root_path = 'data/'
img_suffix = '.png'
seg_map_suffix = '.png'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'
tgt_img_dir = os.path.join(root_path, 'images/train/')
tgt_mask_dir = os.path.join(root_path, 'masks/train/')
os.system('mkdir -p ' + tgt_img_dir)
os.system('mkdir -p ' + tgt_mask_dir)


def read_single_array_from_pil(path):
    return np.asarray(Image.open(path))


def save_png_from_array(arr, save_path, mode=None):
    Image.fromarray(arr, mode=mode).save(save_path)


def convert_label(img, convert_dict):
    arr = np.zeros_like(img, dtype=np.uint8)
    for c, i in convert_dict.items():
        arr[img == c] = i
    return arr


patients_dir = os.path.join(
    root_path, 'Cranium/computed-tomography-images-for-' +
    'intracranial-hemorrhage-detection-and-segmentation-1.0.0' +
    '/Patients_CT')

patients = sorted(os.listdir(patients_dir))
for p in patients:
    data_dir = os.path.join(patients_dir, p, 'brain')
    file_names = os.listdir(data_dir)
    img_w_mask_names = [
        _.replace('_HGE_Seg', '') for _ in file_names if 'Seg' in _
    ]
    img_wo_mask_names = [
        _ for _ in file_names if _ not in img_w_mask_names and 'Seg' not in _
    ]

    for file_name in file_names:
        path = os.path.join(data_dir, file_name)
        img = read_single_array_from_pil(path)
        tgt_name = file_name.replace('.jpg', img_suffix)
        tgt_name = p + '_' + tgt_name
        if 'Seg' in file_name:  # is a mask
            tgt_name = tgt_name.replace('_HGE_Seg', '')
            mask_path = os.path.join(tgt_mask_dir, tgt_name)
            mask = convert_label(img, convert_dict={0: 0, 255: 1})
            save_png_from_array(mask, mask_path)
        else:
            img_path = os.path.join(tgt_img_dir, tgt_name)
            pil = Image.fromarray(img).convert('RGB')
            pil.save(img_path)

            if file_name in img_wo_mask_names:
                mask = np.zeros_like(img, dtype=np.uint8)
                mask_path = os.path.join(tgt_mask_dir, tgt_name)
                save_png_from_array(mask, mask_path)
