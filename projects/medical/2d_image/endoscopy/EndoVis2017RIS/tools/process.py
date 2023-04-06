# """
# [1] Merge masks with different instruments into one binary mask
# [2] Crop black borders from images and masks
# """
# from pathlib import Path

# from tqdm import tqdm
# import cv2
# import numpy as np

# data_path = Path('data')

# train_path = data_path / 'train'

# cropped_train_path = data_path / 'cropped_train'

# original_height, original_width = 1080, 1920
# height, width = 1024, 1280
# h_start, w_start = 28, 320

# binary_factor = 255
# parts_factor = 85
# instrument_factor = 32

# if __name__ == '__main__':
#     for instrument_index in range(1, 9):
#         instrument_folder = 'instrument_dataset_' + str(instrument_index)

#         (cropped_train_path / instrument_folder / 'images').mkdir(exist_ok=True, parents=True)  # noqa

#         binary_mask_folder = (cropped_train_path / instrument_folder / 'binary_masks')  # noqa
#         binary_mask_folder.mkdir(exist_ok=True, parents=True)

#         parts_mask_folder = (cropped_train_path / instrument_folder / 'parts_masks')  # noqa
#         parts_mask_folder.mkdir(exist_ok=True, parents=True)

#         instrument_mask_folder = (cropped_train_path / instrument_folder / 'instruments_masks')  # noqa
#         instrument_mask_folder.mkdir(exist_ok=True, parents=True)

#         mask_folders = list((train_path / instrument_folder / 'ground_truth').glob('*'))  # noqa
#         # mask_folders = [x for x in mask_folders if 'Other' not in str(mask_folders)]  # noqa

#         for file_name in tqdm(list((train_path / instrument_folder / 'left_frames').glob('*'))):  # noqa
#             img = cv2.imread(str(file_name))
#             old_h, old_w, _ = img.shape

#             img = img[h_start: h_start + height, w_start: w_start + width]
#             cv2.imwrite(str(cropped_train_path / instrument_folder / 'images' / (file_name.stem + '.jpg')), img,  # noqa
#                         [cv2.IMWRITE_JPEG_QUALITY, 100])

#             mask_binary = np.zeros((old_h, old_w))
#             mask_parts = np.zeros((old_h, old_w))
#             mask_instruments = np.zeros((old_h, old_w))

#             for mask_folder in mask_folders:
#                 mask = cv2.imread(str(mask_folder / file_name.name), 0)

#                 if 'Bipolar_Forceps' in str(mask_folder):
#                     mask_instruments[mask > 0] = 1
#                 elif 'Prograsp_Forceps' in str(mask_folder):
#                     mask_instruments[mask > 0] = 2
#                 elif 'Large_Needle_Driver' in str(mask_folder):
#                     mask_instruments[mask > 0] = 3
#                 elif 'Vessel_Sealer' in str(mask_folder):
#                     mask_instruments[mask > 0] = 4
#                 elif 'Grasping_Retractor' in str(mask_folder):
#                     mask_instruments[mask > 0] = 5
#                 elif 'Monopolar_Curved_Scissors' in str(mask_folder):
#                     mask_instruments[mask > 0] = 6
#                 elif 'Other' in str(mask_folder):
#                     mask_instruments[mask > 0] = 7

#                 if 'Other' not in str(mask_folder):
#                     mask_binary += mask

#                     mask_parts[mask == 10] = 1  # Shaft
#                     mask_parts[mask == 20] = 2  # Wrist
#                     mask_parts[mask == 30] = 3  # Claspers

#             mask_binary = (mask_binary[h_start: h_start + height, w_start: w_start + width] > 0).astype(  # noqa
#                 np.uint8) * binary_factor
#             mask_parts = (mask_parts[h_start: h_start + height, w_start: w_start + width]).astype(  # noqa
#                 np.uint8) * parts_factor
#             mask_instruments = (mask_instruments[h_start: h_start + height, w_start: w_start + width]).astype(  # noqa
#                 np.uint8) * instrument_factor

#             cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)  # noqa
#             cv2.imwrite(str(parts_mask_folder / file_name.name), mask_parts)  # noqa
#             cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)  # noqa

import argparse
import glob
import os

# from PIL import Image
# import cv2
from sklearn.model_selection import train_test_split

# import shutil

# from tqdm import tqdm

# rp = './data/EndoVis_2017_RIS/Training_dataset/data/cropped_train'
# save_img_path = './data/EndoVis_2017_RIS/images/train'
# save_mask_path = './data/EndoVis_2017_RIS/masks/train'

# for i in range(1, 9):
#     imgs = glob.glob(os.path.join(rp,
#               f'instrument_dataset_{i}', 'images', '*.jpg'))
#     masks = glob.glob(os.path.join(rp,
#               f'instrument_dataset_{i}', 'parts_masks', '*.png'))

#     for imp in tqdm(imgs):
#         shutil.copy(imp, os.path.join(save_img_path,
#       f'dataset{i}_' + os.path.basename(imp).replace('.jpg', '.png')))

#     for maskp in tqdm(masks):
#         mask = cv2.imread(maskp)
#         mask = mask//85
#         cv2.imwrite(os.path.join(save_mask_path,
#           f'dataset{i}_' + os.path.basename(maskp)), mask)

# shutil.copy(maskp, os.path.join(save_mask_path,
#           f'dataset{i}_' + os.path.basename(maskp)))

# rp = './data/EndoVis_2017_RIS/Test_data/data'
# save_img_path = './data/EndoVis_2017_RIS/images/test'
# save_mask_path = './data/EndoVis_2017_RIS/masks/test'

# for i in range(1, 11):
#     imgs = glob.glob(os.path.join(rp, f'instrument_dataset_{i}',
#                               'left_frames', '*.png'))
#     masks = glob.glob(os.path.join(
#                   './data/EndoVis_2017_RIS/Test_labels/instrument_2017_test',
#                   f'instrument_dataset_{i}', 'PartsSegmentation', '*.png'))

#     for imp in tqdm(imgs):
#         shutil.copy(imp, os.path.join(save_img_path, \
#               f'dataset{i}_' + os.path.basename(imp)))

#     for maskp in tqdm(masks):
#         mask = cv2.imread(maskp)
#         mask[mask==30] = 1
#         mask[mask==100] = 2
#         mask[mask==255] = 3
#         cv2.imwrite(os.path.join(save_mask_path, \
#                   f'dataset{i}_' + os.path.basename(maskp)), mask)


def save_anno(img_list, file_path, remove_suffix=True):
    if remove_suffix:  # （文件路径从data/${image/masks}之后的相对路径开始）
        img_list = [
            '/'.join(img_path.split('/')[-2:]) for img_path in img_list
        ]
        img_list = [
            '.'.join(img_path.split('.')[:-1]) for img_path in img_list
        ]
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
    if not os.path.exists(os.path.join(data_root, 'masks/val')) and \
            not os.path.exists(os.path.join(data_root, 'masks/test')):
        all_imgs = sorted(glob.glob(data_root + '/images/train/*.png'))
        x_train, x_val = train_test_split(
            all_imgs, test_size=0.2, random_state=0)
        save_anno(x_train, data_root + '/train.txt')
        save_anno(x_val, data_root + '/val.txt')
    else:
        x_train = sorted(glob.glob(data_root + '/images/train/*.png'))
        save_anno(x_train, data_root + '/train.txt')
    # ------------生成md5值以及包含无标签image的list，pr时该部分代码将被删除-------
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
