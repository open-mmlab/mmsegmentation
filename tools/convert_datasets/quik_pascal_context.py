import argparse
import json
import os
import os.path as osp
from functools import partial

import mmcv
import numpy as np
import scipy.io as io
from PIL import Image

_mapping = np.sort(
    np.array([
        0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 23, 397, 25, 284,
        158, 159, 416, 33, 162, 420, 454, 295, 296, 427, 44, 45, 46, 308, 59,
        440, 445, 31, 232, 65, 354, 424, 68, 326, 72, 458, 34, 207, 80, 355,
        85, 347, 220, 349, 360, 98, 187, 104, 105, 366, 189, 368, 113, 115
    ]))
_key = np.arange(len(_mapping))


def trainval_sets_build(base_folder):
    trainval_json_path = f'{base_folder}/VOC2010/trainval_merged.json'
    context_imagesets_folder = (f'{base_folder}/VOC2010/ImageSets/'
                                'SegmentationContext')

    mmcv.mkdir_or_exist(context_imagesets_folder)

    with open(trainval_json_path, 'r') as fp:
        trainval_json = json.load(fp)

    def regularize_id(int_id):
        str_id = str(int_id)
        return str_id[:4] + '_' + str_id[4:]

    train_list = []
    val_list = []
    images = trainval_json['images']
    for image in images:
        if image['phase'] == 'train':
            train_list.append(regularize_id(image['image_id']))
        elif image['phase'] == 'val':
            val_list.append(regularize_id(image['image_id']))

    # Write train.txt
    with open(f'{context_imagesets_folder}/train.txt', 'w') as fp:
        [fp.write(x + '\n') for x in train_list]
    # Write val.txt
    with open(f'{context_imagesets_folder}/val.txt', 'w') as fp:
        [fp.write(x + '\n') for x in val_list]

    return train_list, val_list


def dconvert_mat_to_segmap(base_folder):
    src_folder = f'{base_folder}/VOC2010/context_raw/trainval/trainval'
    dst_folder = f'{base_folder}/VOC2010/SegmentationClassContext'

    assert osp.exists(src_folder), 'Please put uncompressed \
        raw Pascal Context dataset into context_raw folder.'

    mmcv.mkdir_or_exist(dst_folder)

    mat_list = os.listdir(src_folder)
    for mat_name in mat_list:
        item_name = osp.splitext(mat_name)[0]
        mat_path = osp.join(src_folder, mat_name)
        mat_obj = io.loadmat(mat_path)
        mat_data = mat_obj['LabelMap']
        segmap = np.zeros_like(mat_data, np.uint8)
        for key_index, map_index in zip(_key[1:], _mapping[1:]):
            segmap[mat_data == map_index] = key_index
        segmap = Image.fromarray(segmap)
        segmap.save(osp.join(dst_folder, item_name + '.png'))


def convert_mat_to_segmap(mat_id, src_folder, dst_folder):
    mat_path = osp.join(src_folder, mat_id + '.mat')
    mat_obj = io.loadmat(mat_path)
    mat_data = mat_obj['LabelMap']
    segmap = np.zeros_like(mat_data, np.uint8)
    for key_index, map_index in zip(_key[1:], _mapping[1:]):
        segmap[mat_data == map_index] = key_index
    segmap = Image.fromarray(segmap)
    segmap.save(osp.join(dst_folder, mat_id + '.png'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmsegmentation format')
    parser.add_argument('devkit_path', help='pascal voc devkit path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    base_folder = args.devkit_path
    train_list, val_list = trainval_sets_build(base_folder)

    src_folder = f'{base_folder}/VOC2010/context_raw/trainval/trainval'
    dst_folder = f'{base_folder}/VOC2010/SegmentationClassContext'

    assert osp.exists(src_folder), 'Please put uncompressed \
        raw Pascal Context dataset into context_raw folder.'

    mmcv.mkdir_or_exist(dst_folder)

    mmcv.track_progress(
        partial(
            convert_mat_to_segmap,
            src_folder=src_folder,
            dst_folder=dst_folder), train_list)

    mmcv.track_progress(
        partial(
            convert_mat_to_segmap,
            src_folder=src_folder,
            dst_folder=dst_folder), val_list)

    print('Done!')


if __name__ == '__main__':
    main()
