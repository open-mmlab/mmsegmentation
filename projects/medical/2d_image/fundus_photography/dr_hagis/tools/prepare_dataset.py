import glob
import os
import shutil

import mmengine
import numpy as np
from PIL import Image

root_path = 'data/'
img_suffix = '.jpg'
seg_map_suffix = '_manual_orig.png'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

x_train = glob.glob(os.path.join('data/DRHAGIS/**/*' + img_suffix))

mmengine.mkdir_or_exist(root_path + 'images/train/')
mmengine.mkdir_or_exist(root_path + 'masks/train/')

D3_palette = {0: (0, 0, 0), 1: (1, 1, 1)}
D3_invert_palette = {v: k for k, v in D3_palette.items()}
D2_255_convert_dict = {0: 0, 255: 1}

part_dir_dict = {0: 'train/', 1: 'val/'}
for ith, part in enumerate([x_train]):
    part_dir = part_dir_dict[ith]
    for img in part:
        basename = os.path.basename(img)
        shutil.copy(
            img, root_path + 'images/' + part_dir + basename.split('.')[0] +
            save_img_suffix)
        mask_path = root_path + 'DRHAGIS/Manual_Segmentations/' + basename.split(  # noqa
            '.')[0] + seg_map_suffix
        label = np.array(Image.open(mask_path))

        save_mask_path = root_path + 'masks/' + part_dir + basename.split(
            '.')[0] + save_seg_map_suffix  # noqa
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        mask[mask == 255] = 1
        mask = Image.fromarray(mask)
        mask.save(save_mask_path)
