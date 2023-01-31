import pandas as pd
import random 
from sklearn.model_selection import train_test_split
import os
import shutil
import glob
import mmcv
import numpy as np
from PIL import Image

root_path = "data/Bactteria_Det/"
all_imgs = glob.glob(os.path.join("data/Bactteria_Det/images/*.png"))

x_train, x_test = train_test_split(all_imgs, test_size = 0.2, random_state=0)
print(len(x_train), len(x_test))
os.system("mkdir " + root_path +"images/train/")
os.system("mkdir " + root_path +"images/val/")
os.system("mkdir " + root_path +"masks/train/")
os.system("mkdir " + root_path +"masks/val/")
palette={0: (0, 0, 0), 1: (1, 1, 1)}
invert_palette = {v: k for k, v in palette.items()}

def convert_from_color(arr_3d, palette=invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d
for img in x_train:
    basename = os.path.basename(img)
    shutil.copy(img, root_path +"images/train/"+basename)
    mask_path = root_path +"masks/"+basename
    label = mmcv.imread(mask_path, channel_order='rgb')
    label = convert_from_color(label)
    lab_patch = Image.fromarray(label.astype(np.uint8))
    save_path = root_path +"masks/train/" + basename
    lab_patch.save(save_path)
    cv_img = np.array(Image.open(save_path))
    print(cv_img.shape)


for img in x_test:
    basename = os.path.basename(img)
    shutil.copy(img, root_path +"images/val/"+basename)
    mask_path = root_path +"masks/"+basename
    label = mmcv.imread(mask_path, channel_order='rgb')
    label = convert_from_color(label)
    lab_patch = Image.fromarray(label.astype(np.uint8), mode='P')
    lab_patch.save(root_path +"masks/val/" + basename)
    