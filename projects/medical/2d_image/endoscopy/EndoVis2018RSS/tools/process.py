import argparse
import glob
import os
import shutil

from sklearn.model_selection import train_test_split
# import numpy as np
# from PIL import Image
from tqdm import tqdm

# import sys
# import time

# rp = './data/EndoVis_2018_RSS'
# save_img_path = './data/EndoVis_2018_RSS/images/train'
# save_mask_path = './data/EndoVis_2018_RSS/masks/train'

# # all_colo = []
# for i in range(1, 17):
#     images = glob.glob(os.path.join(rp, 'rss', f'seq_{i}',\
#                               'left_frames', '*.png'))
#     masks = glob.glob(os.path.join(rp, 'rss', f'seq_{i}',\
#                               'labels', '*.png'))

#     for imagep in tqdm(images):
#         shutil.copy(imagep, os.path.join(save_img_path, \
#                           f'seq_{i}_' + os.path.basename(imagep)))

#     # for maskp in tqdm(masks):
#     for maskp in tqdm(masks):
#         mask = Image.open(maskp)
#         mask = mask.convert('L')
#         mask = np.array(mask)
#         # {0, 129, 66, 164, 169, 109, 240, 179, 150, 54, 188}
#         mask[mask == 129] = 1
#         mask[mask == 66] = 2
#         mask[mask == 164] = 3
#         mask[mask == 169] = 4
#         mask[mask == 109] = 5
#         mask[mask == 240] = 6
#         mask[mask == 179] = 7
#         mask[mask == 150] = 8
#         mask[mask == 54] = 9
#         mask[mask == 188] = 10
#         # all_colo.extend(np.unique(mask).tolist())

#         # new_mask = np.zeros((mask.shape[0], mask.shape[1]))
#         # new_mask[((rmask == 0) == (gmask == 0)) == (bmask == 0)] = 0
#         # new_mask[((rmask == 0) == (gmask == 255)) == (bmask == 0)] = 1
#         # new_mask[((rmask == 0) == (gmask == 255)) == (bmask == 255)] = 2
#         # new_mask[((rmask == 125) == (gmask == 255)) == (bmask == 12)] = 3
#         # new_mask[((rmask == 255) == (gmask == 55)) == (bmask == 0)] = 4
#         # new_mask[((rmask == 24) == (gmask == 55)) == (bmask == 125)] = 5
#         # new_mask[((rmask == 187) == (gmask == 155)) == (bmask == 25)] = 6
#         # new_mask[((rmask == 0) == (gmask == 255)) == (bmask == 125)] = 7
#         # new_mask[((rmask == 255) == (gmask == 255)) == (bmask == 125)] = 8
#         # new_mask[((rmask == 123) == (gmask == 15)) == (bmask == 175)] = 9
#         # new_mask[((rmask == 124) == (gmask == 155)) == (bmask == 5)] = 10

#         # # mask[mask == 10] = 1
#         # # mask[mask == 20] = 2
#         # # mask[mask == 255] = 3
#         # print(np.unique(mask[:,:,3]))
#         mask = Image.fromarray(mask)
#         mask.save(os.path.join(save_mask_path, \
#                       f'seq_{i}_' + os.path.basename(maskp)))
#     time.sleep(3)
# # print(set(all_colo))

rp = './data/EndoVis_2018_RSS/Test_data_and_label_release/test_data'
sap = './data/EndoVis_2018_RSS/images/test'

for i in range(1, 4):
    images = glob.glob(os.path.join(rp, f'seq_{i}', 'left_frames', '*.png'))
    # masks = glob.glob(os.path.join(rp, f'seq_{i}','labels', '*.png'))

    for imagep in tqdm(images):
        shutil.copy(imagep,
                    os.path.join(sap, f'seq_{i}_' + os.path.basename(imagep)))


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
    # ----------生成md5值以及包含无标签image的list，pr时该部分代码将被删除----------
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
