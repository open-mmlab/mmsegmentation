import argparse
import glob
import os

# import cv2
# import numpy as np
# import SimpleITK as sitk
# from PIL import Image
from sklearn.model_selection import train_test_split

# import shutil

# from tqdm import tqdm

# rp = './data/QUBIQ2020'
# save_img_path = './data/QUBIQ2020/images/train'
# save_mask_path = './data/QUBIQ2020/masks/train'

# for i in range(1, 4):
#     # 1. read image
#     dirs = glob.glob(os.path.join(rp, f'training_{i}', '*'))
#     for d in tqdm(dirs):
#         d_name = os.path.basename(d)
#         cases = glob.glob(os.path.join(d,'Training', '*'))
#         for c in cases:
#             c_name = os.path.basename(c)
#             img_path = os.path.join(c, 'image.nii.gz')
#             img = sitk.ReadImage(img_path)
#             img = sitk.GetArrayFromImage(img)
#             if len(img.shape) != 2:
#                 continue
#             try:
#                 cv2.imwrite(os.path.join(save_img_path, f'{i}_{d_name}_{c_name}.png'), img)  # noqa
#             except:
#                 continue

#     # 2. read mask
#             mask_path = os.path.join(c, 'task01_seg01.nii.gz')
#             mask = sitk.ReadImage(mask_path)
#             mask = sitk.GetArrayFromImage(mask)
#             if len(mask.shape) != 2:
#                 continue
#             try:
#                 cv2.imwrite(os.path.join(save_mask_path, f'{i}_{d_name}_{c_name}.png'), mask)  # noqa
#             except:
#                 continue

# rp = './data/QUBIQ2020'
# save_img_path = './data/QUBIQ2020/images/val'
# save_mask_path = './data/QUBIQ2020/masks/val'

# for i in range(1, 3):
#     # 1. read image
#     dirs = glob.glob(os.path.join(rp, f'validation_data_{i}', '*'))
#     for d in tqdm(dirs):
#         d_name = os.path.basename(d)
#         cases = glob.glob(os.path.join(d,'Validation', '*'))
#         for c in cases:
#             c_name = os.path.basename(c)
#             img_path = os.path.join(c, 'image.nii.gz')
#             img = sitk.ReadImage(img_path)
#             img = sitk.GetArrayFromImage(img)
#             if len(img.shape) != 2:
#                 continue
#             try:
#                 cv2.imwrite(os.path.join(save_img_path, f'{i}_{d_name}_{c_name}.png'), img)  # noqa
#             except:
#                 continue

#     # 2. read mask
#             mask_path = os.path.join(c, 'task01_seg01.nii.gz')
#             mask = sitk.ReadImage(mask_path)
#             mask = sitk.GetArrayFromImage(mask)
#             if len(mask.shape) != 2:
#                 continue
#             try:
#                 cv2.imwrite(os.path.join(save_mask_path, f'{i}_{d_name}_{c_name}.png'), mask)  # noqa
#             except:
#                 continue


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
