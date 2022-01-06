import argparse
import glob
import os.path as osp
import random
from os import symlink
from shutil import copyfile

import mmcv
import numpy as np

random.seed(14)

# Global variables for specifying label suffix according to class count
LABEL_SUFFIX_19_CLS = '_19LabelTrainIds.png'
LABEL_SUFFIX_34_CLS = '_34LabelTrainIds.png'

# Dictionaries specifying which A2D2 segmentation color corresponds to

# A2D2 'trainId' value
#   key: RGB color, value: trainId
#
# The following segmentation classes are ignored (i.e. trainIds 255):
# - Ego car:      A calibrated system should a priori know what input region
#                 corresponds to the ego vehicle.
# - Blurred area: Ambiguous semantic.
# - Rain dirt:    Ambiguous semantic.
# The following segmentation class is merged:
# - Speed bumper --> RD normal street (50% of dataset contains only one sample)
SEG_COLOR_DICT_34_CLS = {
    (255, 0, 0): 27,  # Car 1
    (200, 0, 0): 27,  # Car 2
    (150, 0, 0): 27,  # Car 3
    (128, 0, 0): 27,  # Car 4
    (182, 89, 6): 26,  # Bicycle 1
    (150, 50, 4): 26,  # Bicycle 2
    (90, 30, 1): 26,  # Bicycle 3
    (90, 30, 30): 26,  # Bicycle 4
    (204, 153, 255): 25,  # Pedestrian 1
    (189, 73, 155): 25,  # Pedestrian 2
    (239, 89, 191): 25,  # Pedestrian 3
    (255, 128, 0): 29,  # Truck 1
    (200, 128, 0): 29,  # Truck 2
    (150, 128, 0): 29,  # Truck 3
    (0, 255, 0): 31,  # Small vehicles 1
    (0, 200, 0): 31,  # Small vehicles 2
    (0, 150, 0): 31,  # Small vehicles 3
    (0, 128, 255): 18,  # Traffic signal 1
    (30, 28, 158): 18,  # Traffic signal 2
    (60, 28, 100): 18,  # Traffic signal 3
    (0, 255, 255): 19,  # Traffic sign 1
    (30, 220, 220): 19,  # Traffic sign 2
    (60, 157, 199): 19,  # Traffic sign 3
    (255, 255, 0): 28,  # Utility vehicle 1
    (255, 255, 200): 28,  # Utility vehicle 2
    (233, 100, 0): 15,  # Sidebars
    (110, 110, 0): 0,  # Speed bumper (*merged due to scarcity)
    (128, 128, 0): 13,  # Curbstone
    (255, 193, 37): 6,  # Solid line
    (64, 0, 64): 21,  # Irrelevant signs
    (185, 122, 87): 16,  # Road blocks
    (0, 0, 100): 30,  # Tractor
    (139, 99, 108): 1,  # Non-drivable street
    (210, 50, 115): 8,  # Zebra crossing
    (255, 0, 128): 33,  # Obstacles / trash
    (255, 246, 143): 17,  # Poles
    (150, 0, 150): 2,  # RD restricted area
    (204, 255, 153): 32,  # Animals
    (238, 162, 173): 9,  # Grid structure
    (33, 44, 177): 20,  # Signal corpus
    (180, 50, 180): 3,  # Drivable cobblestone
    (255, 70, 185): 22,  # Electronic traffic
    (238, 233, 191): 4,  # Slow drive area
    (147, 253, 194): 23,  # Nature object
    (150, 150, 200): 5,  # Parking area
    (180, 150, 200): 12,  # Sidewalk
    (72, 209, 204): 255,  # Ego car <-- IGNORED
    (200, 125, 210): 11,  # Painted driv. instr.
    (159, 121, 238): 10,  # Traffic guide obj.
    (128, 0, 255): 7,  # Dashed line
    (255, 0, 255): 0,  # RD normal street
    (135, 206, 255): 24,  # Sky
    (241, 230, 255): 14,  # Buildings
    (96, 69, 143): 255,  # Blurred area <-- IGNORED
    (53, 46, 82): 255,  # Rain dirt <-- IGNORED
}

# Merged set of segmentation classes used by authors of original paper
SEG_COLOR_DICT_19_CLS = {
    (255, 0, 0): 5,  # Car 1 --> Cars
    (200, 0, 0): 5,  # Car 2 --> Cars
    (150, 0, 0): 5,  # Car 3 --> Cars
    (128, 0, 0): 5,  # Car 4 --> Cars
    (182, 89, 6): 17,  # Bicycle 1 --> Small traffic participants
    (150, 50, 4): 17,  # Bicycle 2 --> Small traffic participants
    (90, 30, 1): 17,  # Bicycle 3 --> Small traffic participants
    (90, 30, 30): 17,  # Bicycle 4 --> Small traffic participants
    (204, 153, 255): 15,  # Pedestrian 1 --> Pedestrians
    (189, 73, 155): 15,  # Pedestrian 2 --> Pedestrians
    (239, 89, 191): 15,  # Pedestrian 3 --> Pedestrians
    (255, 128, 0): 12,  # Truck 1 --> Trucks
    (200, 128, 0): 12,  # Truck 2 --> Trucks
    (150, 128, 0): 12,  # Truck 3 --> Trucks
    (0, 0, 100): 12,  # Tractor --> Trucks
    (0, 255, 0): 17,  # Small vehicles 1 --> Small traffic participants
    (0, 200, 0): 17,  # Small vehicles 2 --> Small traffic participants
    (0, 150, 0): 17,  # Small vehicles 3 --> Small traffic participants
    (0, 128, 255): 9,  # Traffic signal 1 --> Traffic Info
    (30, 28, 158): 9,  # Traffic signal 2 --> Traffic Info
    (60, 28, 100): 9,  # Traffic signal 3 --> Traffic Info
    (0, 255, 255): 9,  # Traffic sign 1 --> Traffic Info
    (30, 220, 220): 9,  # Traffic sign 2 --> Traffic Info
    (60, 157, 199): 9,  # Traffic sign 3 --> Traffic Info
    (255, 255, 0): 12,  # Utility vehicle 1 --> Trucks
    (255, 255, 200): 12,  # Utility vehicle 2 --> Trucks
    (233, 100, 0): 4,  # Sidebars --> Poles
    (110, 110, 0): 1,  # Speed bumper --> Road
    (128, 128, 0): 10,  # Curb stones --> Curb stones
    (255, 193, 37): 6,  # Solid line --> Lane lines
    (64, 0, 64): 8,  # Irrelevant signs --> Irrelevant
    (185, 122, 87): 13,  # Road blocks --> Grid structure
    (139, 99, 108): 11,  # Non-drivable street --> Side walk
    (210, 50, 115): 1,  # Zebra crossing --> Road
    (255, 0, 128): 14,  # Obstacles / trash on road
    (255, 246, 143): 4,  # Poles --> Poles
    (150, 0, 150): 1,  # RD restricted area --> Road
    (204, 255, 153): 15,  # Animals --> Pedestrians
    (238, 162, 173): 13,  # Grid structure --> Grid structure
    (33, 44, 177): 8,  # Signal corpus --> Irrelevant
    (180, 50, 180): 1,  # Drivable cobblestone --> Road
    (255, 70, 185): 9,  # Electronic traffic --> Traffic Info
    (238, 233, 191): 1,  # Slow drive area --> Road
    (147, 253, 194): 3,  # Nature object --> Nature
    (150, 150, 200): 18,  # Parking area
    (180, 150, 200): 11,  # Side walk
    (72, 209, 204): 16,  # Ego car
    (200, 125, 210): 1,  # Painted driv. instr. --> Road
    (159, 121, 238): 9,  # Traffic guide obj. --> Traffic Info
    (128, 0, 255): 6,  # Dashed line --> Lane lines
    (255, 0, 255): 1,  # RD normal street --> Road
    (135, 206, 255): 2,  # Sky
    (241, 230, 255): 7,  # Buildings
    (96, 69, 143): 0,  # Blurred area --> Background
    (53, 46, 82): 255,  # Rain dirt --> IGNORED
}

VAL_SEQS = ['20181008_095521', '20181108_141609', '20181204_154421']
TEST_SEQS = ['20181016_125231', '20181108_084007', '20181204_170238']
# Samples in sequence '20181008_095521' before/after belong to train/val split
SPECIAL_SEQ_ID = '20181008_095521'

SPECIAL_FRAME_SPLIT = 55000


def modify_label_filename(label_filepath, label_choice):
    """Returns a mmsegmentation-combatible label filename."""
    # Ensure that label filenames are modified only once
    if 'TrainIds.png' in label_filepath:
        return label_filepath

    label_filepath = label_filepath.replace('_label_', '_camera_')
    if label_choice == '34_cls':
        label_filepath = label_filepath.replace('.png', LABEL_SUFFIX_34_CLS)
    elif label_choice == '19_cls':
        label_filepath = label_filepath.replace('.png', LABEL_SUFFIX_19_CLS)
    else:
        raise ValueError
    return label_filepath


def convert_34_cls_trainids(label_filepath, ignore_id=255):
    """Saves a new semantic label using 34 class category labels.

    The new image is saved into the same directory as the original image having
    an additional suffix.

    Args:
        label_filepath: Path to the original semantic label.
        ignore_id: Default value for unlabeled elements.
    """
    # Read label file as Numpy array (H, W, 3)
    orig_label = mmcv.imread(label_filepath, channel_order='rgb')

    # Empty array with all elements set as 'ignore id' label
    H, W, _ = orig_label.shape
    mod_label = ignore_id * np.ones((H, W), dtype=int)

    seg_colors = list(SEG_COLOR_DICT_34_CLS.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT_34_CLS[seg_color]

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath, '34_cls')
    label_img = mod_label.astype(np.uint8)
    mmcv.imwrite(label_img, label_filepath)


def convert_19_cls_trainids(label_filepath, ignore_id=255):
    """Saves a new semantic label using 19 class category labels.

    The new image is saved into the same directory as the original image having
    an additional suffix.

    Args:
        label_filepath: Path to the original semantic label.
        ignore_id: Default value for unlabeled elements.
    """
    # Read label file as Numpy array (H, W, 3)
    orig_label = mmcv.imread(label_filepath, channel_order='rgb')

    # Empty array with all elements set as 'ignore id' label
    H, W, _ = orig_label.shape
    mod_label = ignore_id * np.ones((H, W), dtype=int)

    seg_colors = list(SEG_COLOR_DICT_19_CLS.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT_19_CLS[seg_color]

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath, '19_cls')
    label_img = mod_label.astype(np.uint8)
    mmcv.imwrite(label_img, label_filepath)


def create_split_dir(img_filepaths,
                     ann_filepaths,
                     split,
                     root_path,
                     use_symlinks=True):
    """Creates dataset split directory from given file lists using symbolic
    links or copying files.

    Args:
        img_filepaths: List of filepaths as strings.
        ann_filepaths:
        split: String denoting split (i.e. 'train', 'val', or 'test'),
        root_path: A2D2 dataset root directory (.../camera_lidar_semantic/)
        use_symlinks: Symbolically link existing files in the original A2D2
                      dataset directory. If false, files will be copied.

    Raises:
        FileExistError: In case of pre-existing files when trying to create new
                        symbolic links.
    """
    assert split in ['train', 'val', 'test']

    for img_filepath, ann_filepath in zip(img_filepaths, ann_filepaths):
        # Partitions string: [generic/path/to/file] [/] [filename]
        img_filename = img_filepath.rpartition('/')[2]
        ann_filename = ann_filepath.rpartition('/')[2]

        img_link_path = osp.join(root_path, 'images', split, img_filename)
        ann_link_path = osp.join(root_path, 'annotations', split, ann_filename)

        if use_symlinks:
            # NOTE: Can only create new symlinks if no priors ones exists
            try:
                symlink(img_filepath, img_link_path)
            except FileExistsError:
                pass
            try:
                symlink(ann_filepath, ann_link_path)
            except FileExistsError:
                pass

        else:
            copyfile(img_filepath, img_link_path)
            copyfile(ann_filepath, ann_link_path)


def restructure_a2d2_directory(a2d2_path,
                               label_choice,
                               train_on_val_and_test=False,
                               use_symlinks=True):
    """Creates a new directory structure and link existing files into it.

    Required to make the A2D2 dataset conform to the mmsegmentation frameworks
    expected dataset structure.

    my_dataset
    └── images
    │   ├── train
    │   │   ├── xxx{img_suffix}
    |   |   ...
    │   ├── val
    │   │   ├── yyy{img_suffix}
    │   │   ...
    │   ...
    └── annotations
        ├── train
        │   ├── xxx{seg_map_suffix}
        |   ...
        ├── val
        |   ├── yyy{seg_map_suffix}
        |   ...
        ...

    Samples are randomly split into a 'train', 'validation', and 'test' split
    according to the argument sample ratios.

    Args:
        a2d2_path: Absolute path to the A2D2 'camera_lidar_semantic' directory.
        val_ratio: Float value representing ratio of validation samples.
        test_ratio: Float value representing ratio of test samples.
        train_on_val_and_test: Use validation and test samples as training
                               samples if True.
        label_choice: String specifying number of class categories.
        use_symlinks: Symbolically link existing files in the original A2D2
                      dataset directory. If false, files will be copied.
    """

    # Create new directory structure (if not already exist)
    for split in ['train', 'val', 'test']:
        mmcv.mkdir_or_exist(osp.join(a2d2_path, 'images', split))
        mmcv.mkdir_or_exist(osp.join(a2d2_path, 'annotations', split))

    # Lists containing all images and labels to symlinked
    img_filepaths = sorted(glob.glob(osp.join(a2d2_path, '*/camera/*/*.png')))

    if label_choice == '34_cls':
        label_suffix = LABEL_SUFFIX_34_CLS
    elif label_choice == '19_cls':
        label_suffix = LABEL_SUFFIX_19_CLS
    else:
        raise ValueError

    ann_filepaths = sorted(
        glob.glob(osp.join(a2d2_path, '*/label/*/*{}'.format(label_suffix))))

    # Randomize order of (image, label) pairs
    pairs = list(zip(img_filepaths, ann_filepaths))
    random.shuffle(pairs)
    img_filepaths, ann_filepaths = zip(*pairs)

    # Split data according to specified sequences
    train_img_paths = []
    train_ann_paths = []
    val_img_paths = []
    val_ann_paths = []
    test_img_paths = []
    test_ann_paths = []
    for sample_idx in range(len(img_filepaths)):
        img_filepath = img_filepaths[sample_idx]
        ann_filepath = ann_filepaths[sample_idx]

        seq_id = img_filepath.split('/')[-4]

        # NOTE: Need to handle exception where part of a sequence goes to
        # training and the other part to validation split by a frame index
        if seq_id in VAL_SEQS:
            # Parse frame index from last part of filename
            frame_idx = img_filepath.split('_')[-1]
            frame_idx = int(frame_idx.replace('.png', ''))
            # Add sample to validation split unless it is below a frame split
            # index for one of the sequences
            if seq_id == SPECIAL_SEQ_ID and frame_idx < SPECIAL_FRAME_SPLIT:
                # Add sample to train split
                train_img_paths.append(img_filepath)
                train_ann_paths.append(ann_filepath)
            else:
                val_img_paths.append(img_filepath)
                val_ann_paths.append(ann_filepath)
                # To train model on the entire data
                if train_on_val_and_test:
                    train_img_paths = img_filepaths
                    train_ann_paths = ann_filepaths

        elif seq_id in TEST_SEQS:
            # Add sample to test split
            test_img_paths.append(img_filepath)
            test_ann_paths.append(ann_filepath)
            # To train model on the entire data
            if train_on_val_and_test:
                train_img_paths = img_filepaths
                train_ann_paths = ann_filepaths
        else:
            # Add sample to train split
            train_img_paths.append(img_filepath)
            train_ann_paths.append(ann_filepath)

    create_split_dir(
        train_img_paths,
        train_ann_paths,
        'train',
        a2d2_path,
        use_symlinks=use_symlinks)

    create_split_dir(
        val_img_paths,
        val_ann_paths,
        'val',
        a2d2_path,
        use_symlinks=use_symlinks)

    create_split_dir(
        test_img_paths,
        test_ann_paths,
        'test',
        a2d2_path,
        use_symlinks=use_symlinks)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert A2D2 annotations to TrainIds')
    parser.add_argument(
        'a2d2_path',
        help='A2D2 segmentation data absolute path\
                           (NOT the symbolically linked one!)')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument(
        '--no-convert',
        dest='convert',
        action='store_false',
        help='Skips converting label images')
    parser.set_defaults(convert=True)
    parser.add_argument(
        '--no-restruct',
        dest='restruct',
        action='store_false',
        help='Skips restructuring directory structure')
    parser.set_defaults(restruct=True)
    parser.add_argument(
        '--choice',
        default='19_cls',
        help='Label conversion type choice: \'19_cls\' (19 classes) or '
        '\'34_cls\' (34 classes)')
    parser.add_argument(
        '--train-on-val-and-test',
        dest='train_on_val_and_test',
        action='store_true',
        help='Use validation and test samples as training samples')
    parser.set_defaults(train_on_val_and_test=False)
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    parser.add_argument(
        '--no-symlink',
        dest='symlink',
        action='store_false',
        help='Use hard links instead of symbolic links')
    parser.set_defaults(symlink=True)
    args = parser.parse_args()
    return args


def main():
    """A script for making Audi's A2D2 dataset compatible with mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/a2d2)

    Segmentation label conversion:
        The A2D2 labels are instance segmentations (i.e. car_1, car_2, ...),
        while semantic segmentation requires categorical segmentations.

        The function 'convert_CHOICE_trainids()' converts all instance
        segmentation to their corresponding categorical segmentation and saves
        them as new label image files with a filename suffix corresponding to
        the type of class categories.

        Label choice 'cls_19' (default) results in a merged set of 19 classes
        with the filename suffix '_19LabelTrainIds.png'.

        Label choice 'cls_34' results in labels with 34 classes with the
        filename suffix '_34LabelTrainIds.png'.

    The default arguments result in merging of the original 38 semantic classes
    into a 19 class label setup corresponding to the official benchmark results
    presented in the A2D2 paper (ref: p.8 "4. Experiment: Semantic
    segmentation").

    Add `--choice 34_cls` to use the unmerged 34 semantic classes.

    NOTE: The following segmentation classes are ignored (i.e. trainIds 255):
          - Ego car:  A calibrated system should a priori know what input
                      region corresponds to the ego vehicle.
          - Blurred area: Ambiguous semantic.
          - Rain dirt: Ambiguous semantic.

          The following segmentation class is merged due to extreme rarity:
          - Speed bumper --> RD normal street (randomly parsing 50% of dataset
            results in only one sample containing the 'speed_bumper' semantic)

    Training, validation, and test sets are generated according to the same
    sequence split used in the official A2D2 paper benchmark results as
    explained by the authors. The resulting sample count is as follows:
        train | 30699 samples (74.4 %)
        val   |  3246 samples (7.8 %)
        test  |  7332 samples (17.8 %)
        ---------------------
        tot. 41277 samples

    Add the optional argument `--train-on-val-and-test` to train on the entire
    dataset.

    Add `--nproc N` for multiprocessing using N threads.

    Directory restructuring:
        A2D2 files are not natively arranged in the required 'train/val/test'
        directory structure.

        The function 'restructure_a2d2_directory' creates a new compatible
        directory structure in the root directory, The optional argument
        `--no-symlink` creates copies of the label images instead of symbolic
        links.

    Example usage:
        python tools/convert_datasets/a2d2.py path/to/camera_lidar_semantic
    """
    args = parse_args()
    a2d2_path = args.a2d2_path
    out_dir = args.out_dir if args.out_dir else a2d2_path
    mmcv.mkdir_or_exist(out_dir)

    # Create a list of filepaths to all original labels
    # NOTE: Original label files have a number before '.png'
    label_filepaths = glob.glob(osp.join(a2d2_path, '*/label/*/*[0-9].png'))

    # Convert segmentation images to the Cityscapes 'TrainIds' values
    if args.convert:
        seg_choice = args.choice
        if seg_choice == '19_cls':
            if args.nproc > 1:
                mmcv.track_parallel_progress(convert_19_cls_trainids,
                                             label_filepaths, args.nproc)
            else:
                mmcv.track_progress(convert_19_cls_trainids, label_filepaths)
        elif seg_choice == '34_cls':
            if args.nproc > 1:
                mmcv.track_parallel_progress(convert_34_cls_trainids,
                                             label_filepaths, args.nproc)
            else:
                mmcv.track_progress(convert_34_cls_trainids, label_filepaths)
        else:
            raise ValueError

    # Restructure directory structure into 'images' and 'annotations'
    if args.restruct:
        restructure_a2d2_directory(out_dir, args.choice,
                                   args.train_on_val_and_test, args.symlink)


if __name__ == '__main__':
    main()
