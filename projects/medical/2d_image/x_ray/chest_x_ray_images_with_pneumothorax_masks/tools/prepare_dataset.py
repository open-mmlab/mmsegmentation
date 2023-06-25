import glob
import os
import shutil

from PIL import Image
from sklearn.model_selection import train_test_split

root_path = 'data/'
img_suffix = '.png'
seg_map_suffix = '.png'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

all_imgs = glob.glob('data/siim-acr-pneumothorax/png_images/*' + img_suffix)
x_train, x_test = train_test_split(all_imgs, test_size=0.2, random_state=0)

print(len(x_train), len(x_test))
os.system('mkdir -p ' + root_path + 'images/train/')
os.system('mkdir -p ' + root_path + 'images/val/')
os.system('mkdir -p ' + root_path + 'masks/train/')
os.system('mkdir -p ' + root_path + 'masks/val/')

part_dir_dict = {0: 'train/', 1: 'val/'}
for ith, part in enumerate([x_train, x_test]):
    part_dir = part_dir_dict[ith]
    for img in part:
        basename = os.path.basename(img)
        img_save_path = os.path.join(root_path, 'images', part_dir,
                                     basename.split('.')[0] + save_img_suffix)
        shutil.copy(img, img_save_path)
        mask_path = 'data/siim-acr-pneumothorax/png_masks/' + basename
        mask = Image.open(mask_path).convert('L')
        mask_save_path = os.path.join(
            root_path, 'masks', part_dir,
            basename.split('.')[0] + save_seg_map_suffix)
        mask.save(mask_save_path)
