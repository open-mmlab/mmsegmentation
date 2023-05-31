import argparse
import glob
import math
import os
import os.path as osp

import mmcv
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GID dataset to mmsegmentation format')
    parser.add_argument('dataset_img_path', help='GID images folder path')
    parser.add_argument('dataset_label_path', help='GID labels folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument(
        '-o', '--out_dir', help='output path', default='data/gid')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=256)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


GID_COLORMAP = dict(
    Background=(0, 0, 0),  # 0-背景-黑色
    Building=(255, 0, 0),  # 1-建筑-红色
    Farmland=(0, 255, 0),  # 2-农田-绿色
    Forest=(0, 0, 255),  # 3-森林-蓝色
    Meadow=(255, 255, 0),  # 4-草地-黄色
    Water=(0, 0, 255)  # 5-水-蓝色
)
palette = list(GID_COLORMAP.values())
classes = list(GID_COLORMAP.keys())


# 用列表来存一个 RGB 和一个类别的对应
def colormap2label(palette):
    colormap2label_list = np.zeros(256**3, dtype=np.longlong)
    for i, colormap in enumerate(palette):
        colormap2label_list[(colormap[0] * 256 + colormap[1]) * 256 +
                            colormap[2]] = i
    return colormap2label_list


# 给定那个列表，和vis_png然后生成masks_png
def label_indices(RGB_label, colormap2label_list):
    RGB_label = RGB_label.astype('int32')
    idx = (RGB_label[:, :, 0] * 256 +
           RGB_label[:, :, 1]) * 256 + RGB_label[:, :, 2]
    return colormap2label_list[idx]


def RGB2mask(RGB_label, colormap2label_list):
    mask_label = label_indices(RGB_label, colormap2label_list)
    return mask_label


colormap2label_list = colormap2label(palette)


def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    """Original image of GID dataset is very large, thus pre-processing of them
    is adopted.

    Given fixed clip size and stride size to generate
    clipped image, the intersection　of width and height is determined.
    For example, given one 6800 x 7200 original image, the clip size is
    256 and stride size is 256, thus it would generate 29 x 27 = 783 images
    whose size are all 256 x 256.
    """

    image = mmcv.imread(image_path, channel_order='rgb')
    # image = mmcv.bgr2gray(image)

    h, w, c = image.shape
    clip_size = args.clip_size
    stride_size = args.stride_size

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size
    ymin = y * clip_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ],
                     axis=1)

    if to_label:
        image = RGB2mask(image, colormap2label_list)

    for count, box in enumerate(boxes):
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        img_name = osp.basename(image_path).replace('.tif', '')
        img_name = img_name.replace('_label', '')
        if count % 3 == 0:
            mmcv.imwrite(
                clipped_image.astype(np.uint8),
                osp.join(
                    clip_save_dir.replace('train', 'val'),
                    f'{img_name}_{start_x}_{start_y}_{end_x}_{end_y}.png'))
        else:
            mmcv.imwrite(
                clipped_image.astype(np.uint8),
                osp.join(
                    clip_save_dir,
                    f'{img_name}_{start_x}_{start_y}_{end_x}_{end_y}.png'))
        count += 1


def main():
    args = parse_args()
    """
    According to this paper: https://ieeexplore.ieee.org/document/9343296/
    select 15 images contained in GID, , which cover the whole six
    categories, to generate train set and validation set.

    """

    if args.out_dir is None:
        out_dir = osp.join('data', 'gid')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    src_path_list = glob.glob(os.path.join(args.dataset_img_path, '*.tif'))
    print(f'Find {len(src_path_list)} pictures')

    prog_bar = ProgressBar(len(src_path_list))

    dst_img_dir = osp.join(out_dir, 'img_dir', 'train')
    dst_label_dir = osp.join(out_dir, 'ann_dir', 'train')

    for i, img_path in enumerate(src_path_list):
        label_path = osp.join(
            args.dataset_label_path,
            osp.basename(img_path.replace('.tif', '_label.tif')))

        clip_big_image(img_path, dst_img_dir, args, to_label=False)
        clip_big_image(label_path, dst_label_dir, args, to_label=True)
        prog_bar.update()

    print('Done!')


if __name__ == '__main__':
    main()
