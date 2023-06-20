import glob
import os

import numpy as np
from PIL import Image

root_path = 'data/'
img_suffix = '.tif'
seg_map_suffix = '.TIF'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

x_train = glob.glob(
    os.path.join('data/Breast Cancer Cell Segmentation_datasets/Images/*' +
                 img_suffix))

os.system('mkdir -p ' + root_path + 'images/train/')
os.system('mkdir -p ' + root_path + 'masks/train/')

D2_255_convert_dict = {0: 0, 255: 1}


def convert_2d(img, convert_dict=D2_255_convert_dict):
    arr_2d = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for c, i in convert_dict.items():
        arr_2d[img == c] = i
    return arr_2d


part_dir_dict = {0: 'train/'}
for ith, part in enumerate([x_train]):
    part_dir = part_dir_dict[ith]
    for img in part:
        basename = os.path.basename(img)
        img_save_path = root_path + 'images/' + part_dir + basename.split(
            '.')[0] + save_img_suffix
        Image.open(img).save(img_save_path)
        mask_path = root_path + 'Breast Cancer Cell Segmentation_datasets/Masks/' + '_'.join(  # noqa
            basename.split('_')[:-1]) + seg_map_suffix
        label = np.array(Image.open(mask_path))

        save_mask_path = root_path + 'masks/' + part_dir + basename.split(
            '.')[0] + save_seg_map_suffix
        assert len(label.shape) == 2 and 255 in label and 1 not in label
        mask = convert_2d(label)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask.save(save_mask_path)
