import os

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

root_path = 'data/'
img_suffix = '.dcm'
seg_map_suffix = '.png'
save_img_suffix = '.png'
save_seg_map_suffix = '.png'

x_train = []
for fpath, dirname, fnames in os.walk('data/chestimage_train_datasets'):
    for fname in fnames:
        if fname.endswith('.dcm'):
            x_train.append(os.path.join(fpath, fname))
x_test = []
for fpath, dirname, fnames in os.walk('data/chestimage_test_datasets/'):
    for fname in fnames:
        if fname.endswith('.dcm'):
            x_test.append(os.path.join(fpath, fname))

os.system('mkdir -p ' + root_path + 'images/train/')
os.system('mkdir -p ' + root_path + 'images/test/')
os.system('mkdir -p ' + root_path + 'masks/train/')


def rle_decode(rle, width, height):
    mask = np.zeros(width * height, dtype=np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height, order='F')


part_dir_dict = {0: 'train/', 1: 'test/'}
dict_from_csv = pd.read_csv(
    root_path + 'chestimage_train-rle_datasets.csv', sep=',',
    index_col=0).to_dict()[' EncodedPixels']

for ith, part in enumerate([x_train, x_test]):
    part_dir = part_dir_dict[ith]
    for img in part:
        basename = os.path.basename(img)
        img_id = '.'.join(basename.split('.')[:-1])
        if ith == 0 and (img_id not in dict_from_csv.keys()):
            continue
        image = pydicom.read_file(img).pixel_array
        save_img_path = root_path + 'images/' + part_dir + '.'.join(
            basename.split('.')[:-1]) + save_img_suffix
        print(save_img_path)
        img_h, img_w = image.shape[:2]
        image = Image.fromarray(image)
        image.save(save_img_path)
        if ith == 1:
            continue
        if dict_from_csv[img_id] == '-1':
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
        else:
            mask = rle_decode(dict_from_csv[img_id], img_h, img_w)
        save_mask_path = root_path + 'masks/' + part_dir + '.'.join(
            basename.split('.')[:-1]) + save_seg_map_suffix
        mask = Image.fromarray(mask)
        mask.save(save_mask_path)
