import argparse
import os
import shutil

# select 15 images from GID dataset

img_list = [
    'GF2_PMS1__L1A0000647767-MSS1.tif', 'GF2_PMS1__L1A0001064454-MSS1.tif',
    'GF2_PMS1__L1A0001348919-MSS1.tif', 'GF2_PMS1__L1A0001680851-MSS1.tif',
    'GF2_PMS1__L1A0001680853-MSS1.tif', 'GF2_PMS1__L1A0001680857-MSS1.tif',
    'GF2_PMS1__L1A0001757429-MSS1.tif', 'GF2_PMS2__L1A0000607681-MSS2.tif',
    'GF2_PMS2__L1A0000635115-MSS2.tif', 'GF2_PMS2__L1A0000658637-MSS2.tif',
    'GF2_PMS2__L1A0001206072-MSS2.tif', 'GF2_PMS2__L1A0001471436-MSS2.tif',
    'GF2_PMS2__L1A0001642620-MSS2.tif', 'GF2_PMS2__L1A0001787089-MSS2.tif',
    'GF2_PMS2__L1A0001838560-MSS2.tif'
]

labels_list = [
    'GF2_PMS1__L1A0000647767-MSS1_label.tif',
    'GF2_PMS1__L1A0001064454-MSS1_label.tif',
    'GF2_PMS1__L1A0001348919-MSS1_label.tif',
    'GF2_PMS1__L1A0001680851-MSS1_label.tif',
    'GF2_PMS1__L1A0001680853-MSS1_label.tif',
    'GF2_PMS1__L1A0001680857-MSS1_label.tif',
    'GF2_PMS1__L1A0001757429-MSS1_label.tif',
    'GF2_PMS2__L1A0000607681-MSS2_label.tif',
    'GF2_PMS2__L1A0000635115-MSS2_label.tif',
    'GF2_PMS2__L1A0000658637-MSS2_label.tif',
    'GF2_PMS2__L1A0001206072-MSS2_label.tif',
    'GF2_PMS2__L1A0001471436-MSS2_label.tif',
    'GF2_PMS2__L1A0001642620-MSS2_label.tif',
    'GF2_PMS2__L1A0001787089-MSS2_label.tif',
    'GF2_PMS2__L1A0001838560-MSS2_label.tif'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='From 150 images of GID dataset to select 15 images')
    parser.add_argument('dataset_img_dir', help='150 GID images folder path')
    parser.add_argument('dataset_label_dir', help='150 GID labels folder path')

    parser.add_argument('dest_img_dir', help='15 GID images folder path')
    parser.add_argument('dest_label_dir', help='15 GID labels folder path')

    args = parser.parse_args()

    return args


def main():
    """This script is used to select 15 images from GID dataset, According to
    paper: https://ieeexplore.ieee.org/document/9343296/"""
    args = parse_args()

    img_path = args.dataset_img_dir
    label_path = args.dataset_label_dir

    dest_img_dir = args.dest_img_dir
    dest_label_dir = args.dest_label_dir

    # copy images of 'img_list' to 'desr_dir'
    print('Copy images of img_list to desr_dir ing...')
    for img in img_list:
        shutil.copy(os.path.join(img_path, img), dest_img_dir)
    print('Done!')

    print('copy labels of labels_list to desr_dir ing...')
    for label in labels_list:
        shutil.copy(os.path.join(label_path, label), dest_label_dir)
    print('Done!')


if __name__ == '__main__':
    main()
