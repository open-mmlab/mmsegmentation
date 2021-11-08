import argparse
import os.path as osp
import shutil
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from scipy.io import loadmat

COCO_LEN = 10000

clsID_to_trID = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    25: 24,
    27: 25,
    28: 26,
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    44: 40,
    46: 41,
    47: 42,
    48: 43,
    49: 44,
    50: 45,
    51: 46,
    52: 47,
    53: 48,
    54: 49,
    55: 50,
    56: 51,
    57: 52,
    58: 53,
    59: 54,
    60: 55,
    61: 56,
    62: 57,
    63: 58,
    64: 59,
    65: 60,
    67: 61,
    70: 62,
    72: 63,
    73: 64,
    74: 65,
    75: 66,
    76: 67,
    77: 68,
    78: 69,
    79: 70,
    80: 71,
    81: 72,
    82: 73,
    84: 74,
    85: 75,
    86: 76,
    87: 77,
    88: 78,
    89: 79,
    90: 80,
    92: 81,
    93: 82,
    94: 83,
    95: 84,
    96: 85,
    97: 86,
    98: 87,
    99: 88,
    100: 89,
    101: 90,
    102: 91,
    103: 92,
    104: 93,
    105: 94,
    106: 95,
    107: 96,
    108: 97,
    109: 98,
    110: 99,
    111: 100,
    112: 101,
    113: 102,
    114: 103,
    115: 104,
    116: 105,
    117: 106,
    118: 107,
    119: 108,
    120: 109,
    121: 110,
    122: 111,
    123: 112,
    124: 113,
    125: 114,
    126: 115,
    127: 116,
    128: 117,
    129: 118,
    130: 119,
    131: 120,
    132: 121,
    133: 122,
    134: 123,
    135: 124,
    136: 125,
    137: 126,
    138: 127,
    139: 128,
    140: 129,
    141: 130,
    142: 131,
    143: 132,
    144: 133,
    145: 134,
    146: 135,
    147: 136,
    148: 137,
    149: 138,
    150: 139,
    151: 140,
    152: 141,
    153: 142,
    154: 143,
    155: 144,
    156: 145,
    157: 146,
    158: 147,
    159: 148,
    160: 149,
    161: 150,
    162: 151,
    163: 152,
    164: 153,
    165: 154,
    166: 155,
    167: 156,
    168: 157,
    169: 158,
    170: 159,
    171: 160,
    172: 161,
    173: 162,
    174: 163,
    175: 164,
    176: 165,
    177: 166,
    178: 167,
    179: 168,
    180: 169,
    181: 170,
    182: 171
}


def convert_to_trainID(tuple_path, in_img_dir, in_ann_dir, out_img_dir,
                       out_mask_dir, is_train):
    imgpath, maskpath = tuple_path
    shutil.copyfile(
        osp.join(in_img_dir, imgpath),
        osp.join(out_img_dir, 'train2014', imgpath) if is_train else osp.join(
            out_img_dir, 'test2014', imgpath))
    annotate = loadmat(osp.join(in_ann_dir, maskpath))
    mask = annotate['S'].astype(np.uint8)
    mask_copy = mask.copy()
    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID
    seg_filename = osp.join(out_mask_dir, 'train2014',
                            maskpath.split('.')[0] +
                            '_labelTrainIds.png') if is_train else osp.join(
                                out_mask_dir, 'test2014',
                                maskpath.split('.')[0] + '_labelTrainIds.png')
    Image.fromarray(mask_copy).save(seg_filename, 'PNG')


def generate_coco_list(folder):
    train_list = osp.join(folder, 'imageLists', 'train.txt')
    test_list = osp.join(folder, 'imageLists', 'test.txt')
    train_paths = []
    test_paths = []

    with open(train_list) as f:
        for filename in f:
            basename = filename.strip()
            imgpath = basename + '.jpg'
            maskpath = basename + '.mat'
            train_paths.append((imgpath, maskpath))

    with open(test_list) as f:
        for filename in f:
            basename = filename.strip()
            imgpath = basename + '.jpg'
            maskpath = basename + '.mat'
            test_paths.append((imgpath, maskpath))

    return train_paths, test_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description=\
        'Convert COCO Stuff 10k annotations to mmsegmentation format')  # noqa
    parser.add_argument('coco_path', help='coco stuff path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--nproc', default=16, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_path = args.coco_path
    nproc = args.nproc

    out_dir = args.out_dir or coco_path
    out_img_dir = osp.join(out_dir, 'images')
    out_mask_dir = osp.join(out_dir, 'annotations')

    mmcv.mkdir_or_exist(osp.join(out_img_dir, 'train2014'))
    mmcv.mkdir_or_exist(osp.join(out_img_dir, 'test2014'))
    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'train2014'))
    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'test2014'))

    train_list, test_list = generate_coco_list(coco_path)
    assert (len(train_list) +
            len(test_list)) == COCO_LEN, 'Wrong length of list {} & {}'.format(
                len(train_list), len(test_list))

    if args.nproc > 1:
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                in_img_dir=osp.join(coco_path, 'images'),
                in_ann_dir=osp.join(coco_path, 'annotations'),
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=True),
            train_list,
            nproc=nproc)
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                in_img_dir=osp.join(coco_path, 'images'),
                in_ann_dir=osp.join(coco_path, 'annotations'),
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=False),
            test_list,
            nproc=nproc)
    else:
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                in_img_dir=osp.join(coco_path, 'images'),
                in_ann_dir=osp.join(coco_path, 'annotations'),
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=True), train_list)
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                in_img_dir=osp.join(coco_path, 'images'),
                in_ann_dir=osp.join(coco_path, 'annotations'),
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=False), test_list)

    print('Done!')


if __name__ == '__main__':
    main()
