import argparse
import os
import os.path as osp
import tempfile
import zipfile
import numpy as np
import cv2
import mmcv
from PIL import Image
import imgviz

yiliao_palette = {0: (0,0,0),  # no ill(black)
                  1: (255,255,255)  # ill(white)
                  }

yiliao_invert_palette = {v: k for k, v in yiliao_palette.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert 2021“SEED”第二届江苏大数据开发与应用大赛（华录杯）——医疗卫生赛道 dataset '
                    'to mmsegmentation format')
    parser.add_argument(
        '--training_path', help='the training part of 医疗 dataset',
        default='/mnt/data/yjh/datasets/train.zip')
    parser.add_argument(
        '--testing_path', help='the testing part of 医疗 dataset',
        default='/mnt/data/yjh/datasets/test.zip')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path',
                        default='/mnt/data/yjh/datasets/yiliao')
    args = parser.parse_args()
    return args


# 医疗图像
def convert_from_color(arr_3d, palette=yiliao_invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def main():
    args = parse_args()
    training_path = args.training_path
    testing_path = args.testing_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'yiliao')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        print('Extracting training.zip...')
        zip_file = zipfile.ZipFile(training_path)
        zip_file.extractall(tmp_dir)

        print('Generating training dataset...')
        # now_dir = osp.join(tmp_dir, 'train', 'train_org_image')
        # for img_name in os.listdir(now_dir):
        #     img = mmcv.imread(osp.join(now_dir, img_name))
        #     mmcv.imwrite(
        #         img,
        #         osp.join(
        #             out_dir, 'images', 'training',
        #             osp.splitext(img_name)[0] +
        #             '.png'))

        now_dir = osp.join(tmp_dir, 'train', 'train_mask')
        for img_name in os.listdir(now_dir):
            img = mmcv.imread(osp.join(now_dir, img_name))
            img = convert_from_color(img)
            img = Image.fromarray(img.astype(np.uint8), mode="P")
            print(np.unique(img))

            colormap = imgviz.label_colormap(n_label=256)
            colormap[0, :] = [0, 0, 0]
            colormap[1, :] = [255, 255, 255]

            img.putpalette(colormap.flatten())
            save_path = osp.join(out_dir, 'annotations', 'training',
                                 osp.splitext(img_name)[0] + '.png')
            img.save(save_path)

        print('Extracting test.zip...')
        zip_file = zipfile.ZipFile(testing_path)
        zip_file.extractall(tmp_dir)

        print('Generating validation dataset...')
        now_dir = osp.join(tmp_dir, 'test', 'test_org_image')
        for img_name in os.listdir(now_dir):
            img = mmcv.imread(osp.join(now_dir, img_name))
            mmcv.imwrite(
                img,
                osp.join(
                    out_dir, 'images', 'validation',
                    osp.splitext(img_name)[0] + '.png'))

        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
