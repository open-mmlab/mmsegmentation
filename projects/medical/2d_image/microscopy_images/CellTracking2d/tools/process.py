# import os
# import numpy as np
# import shutil
# from PIL import Image
# import glob
# from tqdm import tqdm

# # rp = './data/CellTracking2d/train/BF-C2DL-HSC'  # noqa
# rp = './data/CellTracking2d'

# for tt in ['train', 'test']:
#     dirs = glob.glob(os.path.join(rp, tt, '*'))
#     for d in dirs:  # d: BF-C2DL-HSC

#         for d0 in ['01', '02']:

#             imgs = glob.glob(os.path.join(d, d0, '*.tif'))

#             masks = glob.glob(os.path.join(d, f'{d0}_ST', 'SEG', '*.tif'))

#             for imp in tqdm(imgs):
#                 img = Image.open(imp)
#                 img = np.array(img)
#                 img = Image.fromarray(img)
#                 if tt == 'train':
#                     img.save(os.path.join('./data/CellTracking2d/images/train/', d.split('/')[-1]+'_'+imp.split('/')[-1].replace('.tif', '.png')))  # noqa
#                 elif tt == 'test':
#                     img.save(os.path.join('./data/CellTracking2d/images/test/', d.split('/')[-1]+'_'+imp.split('/')[-1].replace('.tif', '.png')))  # noqa
#                 # img.save(imp.replace('.tif', '.png'))
#                 # os.remove(imp)

#             for maskp in tqdm(masks):
#                 mask = Image.open(maskp)
#                 mask = np.array(mask)
#                 mask = Image.fromarray(mask)
#                 if tt == 'train':
#                     mask.save(os.path.join('./data/CellTracking2d/masks/train', d.split('/')[-1]+'_'+maskp.split('/')[-1].replace('.tif', '.png').replace('man_seg', 't')))  # noqa
#                 elif tt == 'test':
#                     mask.save(os.path.join('./data/CellTracking2d/masks/test', d.split('/')[-1]+'_'+maskp.split('/')[-1].replace('.tif', '.png').replace('man_seg', 't')))  # noqa
#                 # mask.save(maskp.replace('.tif', '.png').replace('man_seg', 't'))  # noqa
#                 # os.remove(maskp)

import argparse
import glob
import os

from sklearn.model_selection import train_test_split


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
    # ------生成md5值以及包含无标签image的list，pr时该部分代码将被删除----------------
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
