# import os
# import shutil
# import glob
# import numpy as np
# from tqdm import tqdm
# from PIL import Image

# rp = './data/EndoVis_2017_KBD'
# save_img_path = './data/EndoVis_2017_KBD/images/train'
# save_mask_path = './data/EndoVis_2017_KBD/masks/train'

# for i in range(1, 5):
#     images = glob.glob(os.path.join(rp, 'kbd', \
#                   f'kidney_dataset_{i}','left_frames', '*.png'))
#     masks = glob.glob(os.path.join(rp, 'kbd', \
#                   f'kidney_dataset_{i}','ground_truth', '*.png'))

#     for imagep in tqdm(images):
#         shutil.copy(imagep, os.path.join(save_img_path, \
#                       f'kidney_dataset_{i}_' + os.path.basename(imagep)))

#     for maskp in tqdm(masks):
#         mask = np.array(Image.open(maskp))
#         mask[mask == 10] = 1
#         mask[mask == 20] = 2
#         mask[mask == 255] = 3
#         # print(np.unique(mask))
#         mask = Image.fromarray(mask)
#         mask.save(os.path.join(save_mask_path, \
#                           f'kidney_dataset_{i}_' + os.path.basename(maskp)))

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
    if not os.path.exists(os.path.join(data_root, 'masks/val')) \
            and not os.path.exists(os.path.join(data_root, 'masks/test')):
        all_imgs = sorted(glob.glob(data_root + '/images/train/*.png'))
        x_train, x_val = train_test_split(
            all_imgs, test_size=0.2, random_state=0)
        save_anno(x_train, data_root + '/train.txt')
        save_anno(x_val, data_root + '/val.txt')
    else:
        x_train = sorted(glob.glob(data_root + '/images/train/*.png'))
        save_anno(x_train, data_root + '/train.txt')
    # ----------生成md5值以及包含无标签image的list，pr时该部分代码将被删除-------
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
