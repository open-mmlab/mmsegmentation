import os

import tifffile as tiff
from PIL import Image

root_path = 'data/'

image_dir = os.path.join(root_path,
                         '2-PM_Vessel_Dataset/raw/vesselNN_dataset/denoised')
label_dir = os.path.join(root_path,
                         '2-PM_Vessel_Dataset/raw/vesselNN_dataset/labels')
tgt_img_train_dir = os.path.join(root_path, 'images/train/')
tgt_mask_train_dir = os.path.join(root_path, 'masks/train/')
os.system('mkdir -p ' + tgt_img_train_dir)
os.system('mkdir -p ' + tgt_mask_train_dir)


def filter_suffix(src_dir, suffix):
    suffix = '.' + suffix if '.' not in suffix else suffix
    file_names = [_ for _ in os.listdir(src_dir) if _.endswith(suffix)]
    file_paths = [os.path.join(src_dir, _) for _ in file_names]
    return sorted(file_paths), sorted(file_names)


if __name__ == '__main__':

    image_path_list, _ = filter_suffix(image_dir, suffix='tif')
    label_path_list, _ = filter_suffix(label_dir, suffix='.tif')

    for img_path, label_path in zip(image_path_list, label_path_list):
        labels = tiff.imread(label_path)
        images = tiff.imread(img_path)
        assert labels.ndim == 3
        assert images.shape == labels.shape
        name = img_path.split('/')[-1].replace('.tif', '')
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
