import glob
import os
import shutil

import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

root_path = 'data/pcam/'
img_suffix = '.png'
seg_map_suffix = '.mat'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

os.system('mkdir -p ' + root_path + 'images/train/')
os.system('mkdir -p ' + root_path + 'images/val/')
os.system('mkdir -p ' + root_path + 'masks/train/')
os.system('mkdir -p ' + root_path + 'masks/val/')

# gz_files = glob.glob(os.path.join("data/pcam/*" + ".h5"))
# gz_file='data/pcam/camelyonpatch_level_2_split_train_x.h5'
# f = h5py.File(gz_file, "r")
# for key in f.keys():
#     for ith, img in enumerate(f[key]):
#         img = Image.fromarray(img)
#         save_image_path = "data/pcam/images/" + str(ith).zfill(8) +".png"
#         img.save(save_image_path)
gz_file = 'data/pcam/camelyonpatch_level_2_split_train_mask.h5'
f = h5py.File(gz_file, 'r')
for ith, mask in enumerate(f['mask']):
    mask = mask.astype(np.uint8).squeeze()
    mask_png = Image.fromarray(mask)
    save_mask_path = 'data/pcam/masks/' + str(ith).zfill(8) + '.png'
    mask_png.save(save_mask_path)

# x_train = glob.glob(os.path.join("data/pcam/train/Images/*" + img_suffix))
# x_test = glob.glob(os.path.join("data/pcam/val/Images/*" + img_suffix))

all_imgs = glob.glob(os.path.join('data/pcam/images/*' + img_suffix))
x_train, x_test = train_test_split(all_imgs, test_size=0.2, random_state=0)
print(len(x_train), len(x_test))

part_dir_dict = {0: 'train/', 1: 'val/'}
for ith, part in enumerate([x_train, x_test]):
    part_dir = part_dir_dict[ith]
    for img in part:
        basename = os.path.basename(img)
        shutil.copy(
            img, root_path + 'images/' + part_dir + basename.split('.')[0] +
            save_img_suffix)

        mask_path = 'data/pcam/masks/' + basename

        save_mask_path = root_path + 'masks/' + part_dir + basename.split(
            '.')[0] + save_seg_map_suffix
        shutil.copy(mask_path, save_mask_path)
