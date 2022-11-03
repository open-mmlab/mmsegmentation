# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
from collections import namedtuple

import mmcv
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert KITTI annotations to TrainIds')
    parser.add_argument('kitti_path', help='kitti data path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--ratio', default=0.25, type=float, help='test ratio splits')
    args = parser.parse_args()
    return args


# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple('Label', [
    'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances',
    'ignoreInEval', 'color'
])

# COPYING AND PASTING TO DEFINE LABELS HERE
# SORRY

KITTI_LABELS = [
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True,
          (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

#############################################

# DEFINING MAPPING


def create_label_dict(labels_a, labels_b):
    """Creates an array such that dictionary[labels_a_train_id] ===
    labels_b_train_id."""
    print('Creating labels dictionary...')
    dictionary = {}
    for label_a in labels_a:
        matching_labels = [
            label for label in labels_b if label.id == label_a.id
        ]
        label_b = matching_labels[0]
        # Use a.id because that's what the original images are in
        dictionary[label_a.id] = label_b.trainId
    print(dictionary)
    return dictionary


LABELS_MAPPING = create_label_dict(KITTI_LABELS, KITTI_LABELS)

# HELPER FUNCTIONS


def relabel_pixel(pixel):
    # print("Relabeling pixel: ", pixel, LABELS_MAPPING[pixel])
    if (pixel in LABELS_MAPPING.keys()):
        # print("Has mapping...", pixel, LABELS_MAPPING[pixel])
        return LABELS_MAPPING[pixel]
    print('no mapping 0', pixel)
    return 0


def relabel_image(image_path, OUTPUT_DIR):
    """Change every pixel from existing label id to new label id."""
    image = Image.open(image_path, 'r')
    width, height = image.size

    print(image_path)
    print(width, height)

    output = Image.new(image.mode, image.size)
    pixels = list(image.getdata())
    new_pixels = list(map(relabel_pixel, pixels))
    pixdict = {}
    for p in pixels:
        pixdict[p] = 1

    # All the labels present in this image
    print(pixdict.keys())

    output.putdata(new_pixels)

    path, image_name = os.path.split(image_path)
    print(image_path, path, image_name)
    output_file = OUTPUT_DIR + image_name

    print('Saving...', output_file)
    output.save(output_file, 'PNG')


def relabel_batch(INPUT_DIR, OUTPUT_DIR):
    """Change every pixel from existing label id to new label id."""
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith('.png'):
                relabel_image(os.path.join(root, file), OUTPUT_DIR)


def main():
    args = parse_args()
    kitti_path = args.kitti_path
    out_dir = args.out_dir if args.out_dir else kitti_path
    mmcv.mkdir_or_exist(out_dir)

    # create the dir structure as same as cityscapes
    splits = ['train', 'val']

    for split in splits:
        mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', split))
        mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', split))

    training_dir = osp.join(kitti_path, 'training')

    mmcv.mkdir_or_exist(osp.join(training_dir, 'gtFine/'))
    relabel_batch(
        osp.join(training_dir, 'semantic/'), osp.join(training_dir, 'gtFine/'))

    original_dataset = list(
        mmcv.scandir(osp.join(training_dir), recursive=True))

    images_path = sorted([
        osp.join(training_dir, path) for path in original_dataset
        if 'image_2/' in path
    ])

    annotations_path = sorted([
        osp.join(training_dir, path) for path in original_dataset
        if 'gtFine/' in path
    ])

    # split the dataset into train and val
    # and copy the images and annotations to the new dir

    split_ratio = int(len(images_path) * (1 - args.ratio))

    for image in images_path[:split_ratio]:
        shutil.copy(image, osp.join(out_dir, 'img_dir', 'train'))

    for image in images_path[split_ratio:]:
        shutil.copy(image, osp.join(out_dir, 'img_dir', 'val'))

    for ann in annotations_path[:split_ratio]:
        shutil.copy(ann, osp.join(out_dir, 'ann_dir', 'train'))

    for ann in annotations_path[split_ratio:]:
        shutil.copy(ann, osp.join(out_dir, 'ann_dir', 'val'))

    print('Converted KITTI annotations to general format...')


if __name__ == '__main__':
    main()
