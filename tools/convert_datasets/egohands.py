"""
Code adapted from
`https://github.com/molyswu/hand_detection/blob/temp/hand_detection/
egohands_dataset_clean.py`

Used to clean egoHands dataset.
Generates images and annotations in 2 folders:
    * '/data/egohands/images/ - copies images from individual to images
    * '/data/egohands/annotations/ - creates masks  of hands (up to 4 masks)
"""
import argparse
import os
import os.path as osp
import tempfile
import zipfile

import cv2
import mmcv
import numpy as np
import scipy.io as sio
import six.moves.urllib as urllib

EGOHANDS_LEN = 48
TRAINING_LEN = int(EGOHANDS_LEN * 0.60)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert egohands dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='path of egohands_data.zip')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def get_masks_and_images(base_path, dir, out_dir, fold):
    image_path_array = []
    files = []
    for root, dirs, filenames in os.walk(osp.join(base_path, dir)):
        for f in filenames:
            if (f.split(".")[1] == "jpg"):
                img_path = osp.join(base_path, dir, f)
                image_path_array.append(img_path)
                files.append(f)

    # sort image_path_array to ensure its in the low to high order
    # expected in polygon.mat
    zip_list = zip(image_path_array, files)
    sorted_pairs = sorted(zip_list)
    tuples = zip(*sorted_pairs)
    image_path_array, file_name = [list(tuple) for tuple in tuples]

    # Contains segmentation info for each 100 frames of 1 dir
    box_fn = osp.join(base_path, dir, "polygons.mat")
    boxes = sio.loadmat(box_fn)

    # there are 100 of these per folder in the egohands dataset
    polygons = boxes["polygons"][0]

    for pointindex, first in enumerate(polygons):
        img_id = image_path_array[pointindex]
        img = cv2.imread(img_id)

        # Save images in image directory
        dst = osp.join(out_dir, 'images', fold,
                       f'{file_name[pointindex]}')
        print(dst)
        cv2.imwrite(dst, img)

        mask = np.zeros((img.shape[0], img.shape[1]))
        for nr_hand, pointlist in enumerate(first):
            has_hand = False
            pts = np.empty((0, 2), int)
            findex = 0
            for point in pointlist:
                if len(point) == 2:
                    has_hand = True
                    x = int(point[0])
                    y = int(point[1])
                    findex += 1
                    append = np.array([[x, y]])
                    pts = np.append(pts, append, axis=0)

            # Fill polynomials around hands
            if has_hand:
                mask = cv2.fillPoly(mask, [pts], nr_hand+1)

        # Save masks in annotation directory
        dst = osp.join(out_dir, 'annotations', fold,
                       f'{file_name[pointindex]}')
        cv2.imwrite(dst, mask)


def generate_derivatives(image_dir, out_dir):
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs[:TRAINING_LEN]:
            get_masks_and_images(image_dir, dir, out_dir, 'training')
        for dir in dirs[TRAINING_LEN:]:
            get_masks_and_images(image_dir, dir, out_dir, 'validation')


# rename image files so we dont have overlapping names
def rename_files(image_dir, rename_file=True):
    if rename_file:
        print("Renaming files")
        loop_index = 0
        for root, dirs, filenames in os.walk(image_dir):
            for dir in dirs:
                for f in os.listdir(osp.join(image_dir, dir)):
                    if (dir not in f):
                        if f.split(".")[1] == "jpg":
                            loop_index += 1
                            src = osp.join(image_dir, dir, f)
                            dst = osp.join(image_dir, dir, f'{dir}_{f}')
                            os.rename(src, dst)
                    else:
                        break


def download_egohands_dataset(dataset_path):
    EGOHANDS_DATASET_URL = \
        "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"

    is_downloaded = os.path.exists(dataset_path)
    if not is_downloaded:
        print(
            "> downloading egohands dataset. This may take a while \
            (1.3GB, say 3-5mins). Coffee break?"
        )
        opener = urllib.request.URLopener()
        opener.retrieve(EGOHANDS_DATASET_URL, dataset_path)
        print("> download complete")


def main():

    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'egohands')
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

    # download egohands_data.zip if not present
    download_egohands_dataset(dataset_path)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        print('Extracting egohands_data.zip...')
        zip_file = zipfile.ZipFile(dataset_path)
        zip_file.extractall(tmp_dir)

        print(f"using tmp_dir {tmp_dir}")
        print('Generating training dataset...')
        work_dir = osp.join(tmp_dir, "_LABELLED_SAMPLES")
        print(f"using work_dir {work_dir}")

        assert len(os.listdir(work_dir)) == EGOHANDS_LEN, \
            'len(os.listdir(work_dir)) != {}'.format(EGOHANDS_LEN)

        rename_files(work_dir)
        generate_derivatives(work_dir, out_dir)
        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
