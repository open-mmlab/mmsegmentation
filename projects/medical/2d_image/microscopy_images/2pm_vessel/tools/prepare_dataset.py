import os

import tifffile as tiff
from PIL import Image

root_path = 'data/'

src_dir = os.path.join(root_path, '2-PM_Vessel_Dataset')
tgt_img_train_dir = os.path.join(root_path, 'images/train/')
tgt_mask_train_dir = os.path.join(root_path, 'masks/train/')
os.system('mkdir -p ' + tgt_img_train_dir)
os.system('mkdir -p ' + tgt_mask_train_dir)


def filter_suffix(src_dir, suffix):
    suffix = '.' + suffix if '.' not in suffix else suffix
    file_names = [_ for _ in os.listdir(src_dir) if _.endswith(suffix)]
    file_paths = [os.path.join(src_dir, _) for _ in file_names]
    return sorted(file_paths), sorted(file_names)


path_list, _ = filter_suffix(src_dir, suffix='.tif')

for path_label in path_list:
    labels = tiff.imread(path_label)
    assert labels.ndim == 3
    path_image = path_label.replace('_label', '')
    name = path_image.split('/')[-1].replace('.tif', '')
    images = tiff.imread(path_image)
    assert images.shape == labels.shape
    # a single .tif file contains multiple slices
    # as long as it is read by tifffile package.
    for i in range(labels.shape[0]):
        slice_name = name + '_' + str(i).rjust(3, '0') + '.png'
        image = images[i]
        label = labels[i] // 255

        save_path_label = os.path.join(tgt_mask_train_dir, slice_name)
        Image.fromarray(label).save(save_path_label)
        save_path_image = os.path.join(tgt_img_train_dir, slice_name)
        Image.fromarray(image).convert('RGB').save(save_path_image)
