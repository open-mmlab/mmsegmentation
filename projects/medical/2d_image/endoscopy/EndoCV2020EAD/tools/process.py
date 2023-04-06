# import os
# import shutil
# import glob
# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# rp = './data/EndoCV2020_EAD/EAD2'
# save_img_path = './data/EndoCV2020_EAD/images/train'
# save_mask_path = './data/EndoCV2020_EAD/masks/train'

# images = glob.glob(os.path.join(rp, 'images', '*.jpg'))
# masks = glob.glob(os.path.join(rp, 'masks', '*.tif'))

# for imagep in tqdm(images):
#     image_name = os.path.basename(imagep)
#     shutil.copy(imagep, os.path.join(save_img_path, \
#                 image_name.replace('.jpg', '.png')))

# for maskp in tqdm(masks):
# # for maskp in masks:
#     mask = Image.open(maskp)
#     mask = np.array(mask)
#     # print(np.unique(mask))
#     mask[mask < 128] = 0
#     mask[mask >=128] = 1
#     mask = Image.fromarray(mask)
#     mask_name = os.path.basename(maskp)
#     mask.save(os.path.join(save_mask_path, \
#               mask_name.replace('_mask.tif', '.png')))

import argparse
import glob
import os

from sklearn.model_selection import train_test_split


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
    if not os.path.exists(os.path.join(
            data_root, 'masks/val')) and not os.path.exists(
                os.path.join(data_root, 'masks/test')):
        all_imgs = sorted(glob.glob(data_root + '/images/train/*.png'))
        x_train, x_val = train_test_split(
            all_imgs, test_size=0.2, random_state=0)
        save_anno(x_train, data_root + '/train.txt')
        save_anno(x_val, data_root + '/val.txt')
    else:
        x_train = sorted(glob.glob(data_root + '/images/train/*.png'))
        save_anno(x_train, data_root + '/train.txt')
    # ---------------生成md5值以及包含无标签image的list，pr时该部分代码将被删除----------
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
