# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import tempfile
import zipfile

import mmcv
from mmengine.utils import mkdir_or_exist

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MICCAI dataset to mmsegmentation format"
    )
    parser.add_argument("dataset_path", help="path of MICCAI train.zip)")
    parser.add_argument("--tmp_dir", help="path of the temporary directory")
    parser.add_argument("-o", "--out_dir", default=None, help="output path")
    parser.add_argument(
        "-s", "--split_rate", type=float, help="splite rate for train val"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join("data", "miccai")
    else:
        out_dir = args.out_dir

    print("Making directories...")
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, "images"))
    mkdir_or_exist(osp.join(out_dir, "images", "training"))
    mkdir_or_exist(osp.join(out_dir, "images", "validation"))
    mkdir_or_exist(osp.join(out_dir, "annotations"))
    mkdir_or_exist(osp.join(out_dir, "annotations", "training"))
    mkdir_or_exist(osp.join(out_dir, "annotations", "validation"))

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        print("Extracting train.zip...")
        zip_file = zipfile.ZipFile(dataset_path)
        zip_file.extractall(tmp_dir)

        # split
        train_img_path = osp.join(tmp_dir, "train/image")
        train_anno_path = osp.join(tmp_dir, "train/mask")
        image_num = len(os.listdir(train_img_path))
        training_len = int(image_num * args.split_rate)

        print("Processing train data...")

        for img_name in tqdm(sorted(os.listdir(train_img_path))[:training_len]):
            img = mmcv.imread(osp.join(train_img_path, img_name))
            mmcv.imwrite(
                img,
                osp.join(
                    out_dir,
                    "images",
                    "training",
                    osp.splitext(img_name)[0] + ".png",
                ),
            )
        print("Processing train anno data...")
        for img_name in tqdm(sorted(os.listdir(train_anno_path))[:training_len]):
            # The annotation img should be divided by 128, because some of
            # the annotation imgs are not standard. We should set a
            # threshold to convert the nonstandard annotation imgs. The
            # value divided by 128 is equivalent to '1 if value >= 128
            # else 0'
            img = mmcv.imread(osp.join(train_anno_path, img_name))
            mmcv.imwrite(
                img[:, :, 0] // 128,
                osp.join(
                    out_dir,
                    "annotations",
                    "training",
                    osp.splitext(img_name)[0] + ".png",
                ),
            )
        print("Processing val data...")
        for img_name in tqdm(sorted(os.listdir(train_img_path))[training_len:]):
            img = mmcv.imread(osp.join(train_img_path, img_name))
            mmcv.imwrite(
                img,
                osp.join(
                    out_dir,
                    "images",
                    "validation",
                    osp.splitext(img_name)[0] + ".png",
                ),
            )
        print("Processing val anno data...")
        for img_name in tqdm(sorted(os.listdir(train_anno_path))[training_len:]):
            img = mmcv.imread(osp.join(train_anno_path, img_name))
            mmcv.imwrite(
                img[:, :, 0] // 128,
                osp.join(
                    out_dir,
                    "annotations",
                    "validation",
                    osp.splitext(img_name)[0] + ".png",
                ),
            )
        print("Removing the temporary files...")
    print("Done!")


if __name__ == "__main__":
    main()
