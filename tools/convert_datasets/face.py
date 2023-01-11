# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CelebAMask-HQ dataset to mmsegmentation format')
    parser.add_argument('--dataset_dir', help='CelebAMask-HQ folder path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    dataset_dir = Path(args.dataset_dir)
    image_out_dir = out_dir / 'img'
    mask_out_dir = out_dir / 'mask'
    out_dir.mkdir(exist_ok=True)
    image_out_dir.mkdir(exist_ok=True)
    mask_out_dir.mkdir(exist_ok=True)

    # copy image to data folder
    image_dataset_dir = dataset_dir / 'CelebA-HQ-img'
    for image_file in tqdm(image_dataset_dir.glob('*.jpg')):
        shutil.copyfile(image_file, (image_out_dir / image_file.name))
    print(f'Transferred {image_dataset_dir} to {image_out_dir}')
    # convert mask to label image
    mask_dataset_dir = dataset_dir / 'CelebAMask-HQ-masks_corrected'
    for mask_file in tqdm(mask_dataset_dir.glob('*.png')):
        mask = Image.open(mask_file)
        mask = mask.convert('L')
        mask = np.array(mask).astype(int)
        cv2.imwrite(str(mask_out_dir / mask_file.name), mask)
    print(f'Transferred {mask_dataset_dir} to {mask_out_dir}')


if __name__ == '__main__':
    main()
