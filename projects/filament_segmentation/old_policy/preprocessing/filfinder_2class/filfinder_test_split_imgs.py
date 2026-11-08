# https://qiita.com/Shinji_Fujita/items/856010bead47ff337941


from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
# import aplpyder2D, Filament2D
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import fil_finder
# from fil_finder import FilFin
import tqdm
import copy
import random
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.colorbar import ColorbarBase
import pandas as pd
# import warnings # warningが気になる方はこちらも
# warnings.simplefilter('ignore')

import os
import matplotlib
matplotlib.use('Agg')
import pickle
import time
import tempfile

import pandas as pd
import cv2
import glob
import shutil

from PIL import Image
from PIL.Image import Image as PILImage
from tifffile import imwrite

# https://zenn.dev/e4exp/articles/20240411-mmsegm-custom-data

#root_dir = r"E:\Dropbox\AUT\QLab\Airi\data\aquilaM2-250.fits"
#root_dir = r"F:\Dropbox\AUT\QLab\Airi\data\aquilaM2-250.fits"
root_dir = r"/home1/ozaki.a/data"
dst_root_dir = r"/home1/ozaki.a/train_data"

# fits_name = r"aquilaM2-250"
fits_name = r"pipeCenterB59-250"
# fits_name = r"serpens-250"

save_ext = "pdf"
#save_ext = "png"

def convert(img: np.ndarray) -> PILImage:
    palette = [
        [0, 0, 0],
        [255, 0, 0],
    ]

    seg_img = Image.fromarray(img).convert("P")
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    return seg_img

def check_make_dir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


# ファイルがあればPickleとして読み込む（なければNoneを返す）
def load_pkl(fpath):
    if(os.path.exists(fpath)):
        try:
            with open(fpath, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError):
            return None
    return None

# オブジェクトをPickleとして保存
def save_pkl(fpath, val):
    target_dir = os.path.dirname(fpath)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target_dir or None, suffix=".tmp")
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(val, f)
        os.replace(tmp_path, fpath)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


size = 256
# sep_x = 1750 #aquila
sep_x = 1150 #pipeCenter
# sep_x = 2750 #serpens
step_rate = 0.25


def crop_data(nda, nda_check_nan, suf):
    l_dir = os.path.join(root_dir, fits_name, "left")
    r_dir = os.path.join(root_dir, fits_name, "right")
    # serpens_test = os.path.join(root_dir, fits_name, "serpens_test")
    check_make_dir(l_dir)
    check_make_dir(r_dir)
    # check_make_dir(serpens_test)

    w, h = nda.shape[1::-1]
    print(w)
    print(h)
    step = int(step_rate * size)

    y = 0
    while (y + size <= h):
        x = 0
        while (x + size <= w):
            print(str(x) + "," + str(y))
            cropped = nda[y:y + size, x:x + size]
            cropped_check = nda_check_nan[y:y + size, x:x + size]
           # print(cropped)
            if (x + size < sep_x):
                o_dir = l_dir
            else:
                o_dir = r_dir
            # o_dir = serpens_test

            o_name = fits_name + "_" + str(x).zfill(5) + "-" + str(y).zfill(5)

            x += step
            if np.isnan(cropped_check).any():
                continue

            o_path = os.path.join(o_dir, o_name + "_" + suf + ".pkl")
            save_pkl(o_path, cropped)
            o_path = os.path.join(o_dir, o_name + "_" + suf + ".png")
            cv2.imwrite(o_path, cropped.astype("int"))
            # tiff画像の保存
            tiff_path = os.path.join(o_dir, o_name + "_" + suf + ".tif")
            imwrite(tiff_path, cropped.astype(np.float32))

        y += step


def dilate(img):
#    kernel = np.ones((5,5),np.uint8)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    return dilation


# val_sep_y = 1800 #aquila
# test_sep_y = 3018 #aquila
val_sep_y = 700 #pipeCenter
test_sep_y = 1507 #pipeCenter
# val_sep_y = 1800 #serpens
# test_sep_y = 2777 #serpens

train_pos_name = "right"
test_pos_name = "left"

dataset_name = "250115"

train_src_dir = os.path.join(root_dir, fits_name, train_pos_name)
test_src_dir = os.path.join(root_dir, fits_name, test_pos_name)

train_dst_dir = os.path.join(dst_root_dir, fits_name, dataset_name, "train")
val_dst_dir = os.path.join(dst_root_dir, fits_name, dataset_name, "val")
test_dst_dir = os.path.join(dst_root_dir, fits_name, dataset_name, "test")

#dilate_flg = False
dilate_flg = True

# train
# トレーニング用の処理（right）
# paths_pkl_train = glob.glob(os.path.join(val_src_dir, "*_in.png"))
# print(f"Training files: {len(paths_pkl_train)} files found")  # デバッグ用に出力
# for p in paths_pkl_train:
#     fname = os.path.basename(p)
#     size_info = fname.split('_')[1]
#     x_str, y_str = size_info.split('-')
#     x = int(x_str)
#     y = int(y_str)
#
#     # right側のデータはトレーニング用
#     dst_dir = val_dst_dir
#
#     dst_imgs_dir = os.path.join(dst_dir, "imgs")
#     dst_mask_dir = os.path.join(dst_dir, "mask")
#     dst_mask_index_dir = os.path.join(dst_dir, "mask_index")
#
#     check_make_dir(dst_imgs_dir)
#     check_make_dir(dst_mask_dir)
#     check_make_dir(dst_mask_index_dir)
#
#     # 画像をコピーして処理
#     img = cv2.imread(p, 0)
#     dst_img_path = os.path.join(dst_imgs_dir, fname.replace('_in.png', '.png'))
#     cv2.imwrite(dst_img_path, img)
#
#     o_path = p.replace('_in.png', '_out.png')
#     new_path = shutil.copy2(o_path, os.path.join(dst_mask_dir, fname.replace('_in.png', '.png')))
#
#     # 膨張処理とマスクの処理
#     mask = cv2.imread(o_path, 0)
#     if dilate_flg:
#         mask = dilate(mask)
#     print(mask.dtype)
#     mask[mask != 0] = 1
#     mask: PILImage = convert(mask)
#     dst_mi_path = os.path.join(dst_mask_index_dir, fname.replace('_in.png', '.png'))
#     mask.save(dst_mi_path)
#
# # 検証用の処理（left）
# paths_pkl_val = glob.glob(os.path.join(train_src_dir, "*_in.png"))
# print(f"Validation files: {len(paths_pkl_val)} files found")  # デバッグ用に出力
# for p in paths_pkl_val:
#     fname = os.path.basename(p)
#     size_info = fname.split('_')[1]
#     x_str, y_str = size_info.split('-')
#     x = int(x_str)
#     y = int(y_str)
#
#     # left側のデータは検証用
#     dst_dir = train_dst_dir
#
#     dst_imgs_dir = os.path.join(dst_dir, "imgs")
#     dst_mask_dir = os.path.join(dst_dir, "mask")
#     dst_mask_index_dir = os.path.join(dst_dir, "mask_index")
#
#     check_make_dir(dst_imgs_dir)
#     check_make_dir(dst_mask_dir)
#     check_make_dir(dst_mask_index_dir)
#
#     # 画像をコピーして処理
#     img = cv2.imread(p, 0)
#     dst_img_path = os.path.join(dst_imgs_dir, fname.replace('_in.png', '.png'))
#     cv2.imwrite(dst_img_path, img)
#
#     o_path = p.replace('_in.png', '_out.png')
#     new_path = shutil.copy2(o_path, os.path.join(dst_mask_dir, fname.replace('_in.png', '.png')))
#
#     # 膨張処理とマスクの処理
#     mask = cv2.imread(o_path, 0)
#     if dilate_flg:
#         mask = dilate(mask)
#     print(mask.dtype)
#     mask[mask != 0] = 1
#     mask: PILImage = convert(mask)
#     dst_mi_path = os.path.join(dst_mask_index_dir, fname.replace('_in.png', '.png'))
#     mask.save(dst_mi_path)

# train
paths_pkl = glob.glob(os.path.join(train_src_dir, "*_in.png"))
for p in paths_pkl:
    fname = os.path.basename(p)
    size_info = fname.split('_')[1]
    x_str, y_str = size_info.split('-')
    x = int(x_str)
    y = int(y_str)

    # if y < val_sep_y:
    #     dst_dir = train_dst_dir
    # else:
    #     dst_dir = val_dst_dir

    if y < val_sep_y:
        dst_dir = train_dst_dir
    elif val_sep_y <= y < test_sep_y:
        dst_dir = val_dst_dir
    else:
        dst_dir = val_dst_dir

    dst_imgs_dir = os.path.join(dst_dir, "imgs")
    dst_mask_dir = os.path.join(dst_dir, "mask")
    dst_mask_index_dir = os.path.join(dst_dir, "mask_index")

    check_make_dir(dst_imgs_dir)
    check_make_dir(dst_mask_dir)
    check_make_dir(dst_mask_index_dir)

#    new_path = shutil.copy2(p, os.path.join(dst_imgs_dir, fname.replace('_in.png', '.png')))
    img = cv2.imread(p, 0)
#    img = cv2.bitwise_not(img)
    dst_img_path = os.path.join(dst_imgs_dir, fname.replace('_in.png', '.png'))
    cv2.imwrite(dst_img_path, img)

    o_path = p.replace('_in.png', '_out.png')
    new_path = shutil.copy2(o_path, os.path.join(dst_mask_dir, fname.replace('_in.png', '.png')))

    mask = cv2.imread(o_path, 0)
    if dilate_flg:
        mask = dilate(mask)
    print(mask.dtype)
    mask[mask != 0] = 1
    mask: PILImage = convert(mask)
    dst_mi_path = os.path.join(dst_mask_index_dir, fname.replace('_in.png', '.png'))
    mask.save(dst_mi_path)
#    cv2.imwrite(dst_mi_path, mask)

paths_pkl = glob.glob(os.path.join(train_src_dir, "*_in.tif"))
for p in paths_pkl:
    fname = os.path.basename(p)
    size_info = fname.split('_')[1]
    x_str, y_str = size_info.split('-')
    x = int(x_str)
    y = int(y_str)

    if y < val_sep_y:
        dst_dir = train_dst_dir
    elif val_sep_y <= y < test_sep_y:
        dst_dir = val_dst_dir
    else:
        dst_dir = val_dst_dir

    dst_tif_imgs_dir = os.path.join(dst_dir, "tif_imgs")
    check_make_dir(dst_tif_imgs_dir)


    img = cv2.imread(p,-1)
    # print(img)
    dst_img_path = os.path.join(dst_tif_imgs_dir, fname.replace('_in.tif', '.tif'))
    cv2.imwrite(dst_img_path, img)




# test
paths_pkl = glob.glob(os.path.join(test_src_dir, "*_in.png"))
for p in paths_pkl:
    fname = os.path.basename(p)
    size_info = fname.split('_')[1]
    x_str, y_str = size_info.split('-')
    x = int(x_str)
    y = int(y_str)

    dst_imgs_dir = os.path.join(test_dst_dir, "imgs")
    dst_mask_dir = os.path.join(test_dst_dir, "mask")
    dst_mask_index_dir = os.path.join(test_dst_dir, "mask_index")

    check_make_dir(dst_imgs_dir)
    check_make_dir(dst_mask_dir)
    check_make_dir(dst_mask_index_dir)

#    new_path = shutil.copy2(p, os.path.join(dst_imgs_dir, fname.replace('_in.png', '.png')))
    img = cv2.imread(p, 0)
#    img = cv2.bitwise_not(img)
    dst_img_path = os.path.join(dst_imgs_dir, fname.replace('_in.png', '.png'))
    cv2.imwrite(dst_img_path, img)

    o_path = p.replace('_in.png', '_out.png')
    new_path = shutil.copy2(o_path, os.path.join(dst_mask_dir, fname.replace('_in.png', '.png')))

#    mask = load_pkl(o_path)
    mask = cv2.imread(o_path, 0)
    if dilate_flg:
        mask = dilate(mask)
    mask[mask != 0] = 1
    mask: PILImage = convert(mask)
    dst_mi_path = os.path.join(dst_mask_index_dir, fname.replace('_in.png', '.png'))
    mask.save(dst_mi_path)
   # cv2.imwrite(dst_mi_path, mask)

paths_pkl = glob.glob(os.path.join(test_src_dir, "*_in.tif"))
for p in paths_pkl:
    fname = os.path.basename(p)
    size_info = fname.split('_')[1]
    x_str, y_str = size_info.split('-')
    x = int(x_str)
    y = int(y_str)

    dst_tif_imgs_dir = os.path.join(test_dst_dir, "tif_imgs")
    check_make_dir(dst_tif_imgs_dir)

    img = cv2.imread(p, -1)
    dst_img_path = os.path.join(dst_tif_imgs_dir, fname.replace('_in.tif', '.tif'))
    cv2.imwrite(dst_img_path, img)
