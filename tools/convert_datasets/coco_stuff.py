import argparse
import os
import os.path as osp
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from scipy.io import loadmat

COCO_LEN = 10000


def convert_mat(mat_file, in_dir, out_dir):
    data = loadmat(osp.join(in_dir, mat_file))
    mask = data['S'].astype(np.uint8)
    seg_filename = osp.join(out_dir, mat_file.replace('.mat', '.png'))
    Image.fromarray(mask).save(seg_filename, 'PNG')


def generate_coco_list(folder):
    mask_paths = []
    list_path = os.path.join(folder, 'imageLists', 'all.txt')
    with open(list_path) as f:
        for filename in f:
            basename = filename.strip()
            maskpath = basename + '.mat'
            # print(imgpath, maskpath)
            mask_paths.append(maskpath)
    return mask_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert COCO Stuff annotations to mmsegmentation format')
    parser.add_argument('coco_path', help='coco stuff path')
    parser.add_argument('annotations_path', help='annotations path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_path = args.coco_path
    annotations_path = args.annotations_path
    nproc = args.nproc
    if args.out_dir is None:
        out_dir = osp.join(coco_path, 'mask')
    else:
        out_dir = args.out_dir
    mmcv.mkdir_or_exist(out_dir)
    in_dir = osp.join(coco_path, annotations_path)

    all_list = generate_coco_list(coco_path)
    assert len(all_list) == COCO_LEN, 'Wrong length of list {}'.format(
        len(all_list))

    mmcv.track_parallel_progress(
        partial(convert_mat, in_dir=in_dir, out_dir=out_dir),
        all_list,
        nproc=nproc)

    print('Done!')


if __name__ == '__main__':
    main()
