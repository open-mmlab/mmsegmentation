import glob
import os

import numpy as np
from PIL import Image

root_path = 'data/'
img_suffix = '.jpg'
seg_map_suffix_list = ['.jpg', '.png', '.tif']
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

x_train = glob.glob(
    os.path.join('data/Vessels-Datasets/*/Train/Original/Images/*' +
                 img_suffix))
x_test = glob.glob(
    os.path.join('data/Vessels-Datasets/*/Test/Original/Images/*' +
                 img_suffix))

os.system('mkdir -p ' + root_path + 'images/train/')
os.system('mkdir -p ' + root_path + 'images/test/')
os.system('mkdir -p ' + root_path + 'masks/train/')
os.system('mkdir -p ' + root_path + 'masks/test/')

part_dir_dict = {0: 'train/', 1: 'test/'}
for ith, part in enumerate([x_train, x_test]):
    part_dir = part_dir_dict[ith]
    for img in part:
        type_name = img.split('/')[-5]
        basename = type_name + '_' + os.path.basename(img)
        save_img_path = root_path + 'images/' + part_dir + basename.split(
            '.')[0] + save_img_suffix
        Image.open(img).save(save_img_path)

        for seg_map_suffix in seg_map_suffix_list:
            if os.path.exists('/'.join(img.split('/')[:-1]).replace(
                    'Images', 'Labels')):
                mask_path = img.replace('Images', 'Labels').replace(
                    img_suffix, seg_map_suffix)
            else:
                mask_path = img.replace('Images', 'labels').replace(
                    img_suffix, seg_map_suffix)
            if os.path.exists(mask_path):
                break
        save_mask_path = root_path + 'masks/' + part_dir + basename.split(
            '.')[0] + save_seg_map_suffix
        masks = np.array(Image.open(mask_path).convert('L')).astype(np.uint8)
        if len(np.unique(masks)) == 2 and 1 in np.unique(masks):
            print(np.unique(masks))
            pass
        else:
            masks[masks < 128] = 0
            masks[masks >= 128] = 1
        masks = Image.fromarray(masks)
        masks.save(save_mask_path)
