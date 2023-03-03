import glob
import os
import shutil

from PIL import Image

root_path = 'data/'
img_suffix = '.png'
seg_map_suffix = '.png'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

x_train = glob.glob(
    'data/Bacteria_detection_with_darkfield_microscopy_datasets/images/*' +
    img_suffix)  # noqa

os.system('mkdir -p ' + root_path + 'images/train/')
os.system('mkdir -p ' + root_path + 'masks/train/')

part_dir_dict = {0: 'train/'}
for ith, part in enumerate([x_train]):
    part_dir = part_dir_dict[ith]
    for img in part:
        basename = os.path.basename(img)
        img_save_path = os.path.join(root_path, 'images', part_dir,
                                     basename.split('.')[0] + save_img_suffix)
        shutil.copy(img, img_save_path)
        mask_path = 'data/Bacteria_detection_with_darkfield_microscopy_datasets/masks/' + basename  # noqa
        mask = Image.open(mask_path).convert('L')
        mask_save_path = os.path.join(
            root_path, 'masks', part_dir,
            basename.split('.')[0] + save_seg_map_suffix)
        mask.save(mask_save_path)
