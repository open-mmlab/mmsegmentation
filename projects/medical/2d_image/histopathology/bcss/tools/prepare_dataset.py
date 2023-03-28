import glob
import os
import shutil

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

img_folders = (os.path.join(root_path, 'BCSS/rgbs_colorNormalized'),
               os.path.join(root_path, 'BCSS/rgbs_colorNormalized (2)'),
               os.path.join(root_path, 'BCSS/rgbs_colorNormalized (3)'))
x_train = []

for img_folder in img_folders:
    x_train += glob.glob(os.path.join(img_folder, '*' + img_suffix))

for img_path in x_train:
    basename = os.path.basename(img_path)
    img_save_path = os.path.join(root_path, 'images/train',
                                 basename.split('.')[0] + save_img_suffix)
    shutil.copy(img_path, img_save_path)
    mask_path = 'data/BCSS/masks/' + basename
    mask = Image.open(mask_path).convert('L')
    mask_save_path = os.path.join(root_path, 'masks/train',
                                  basename.split('.')[0] + save_seg_map_suffix)
    mask.save(mask_save_path)
