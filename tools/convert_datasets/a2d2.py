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
LABEL_SUFFIX_18_CLS = '_18LabelTrainIds.png'
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
SEG_COLOR_DICT_A2D2 = {
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

# Cityscapes-like 'trainId' value
#   key: RGB color, value: trainId (from Cityscapes)
SEG_COLOR_DICT_CITYSCAPES = {
    (255, 0, 0): 12,  # Car 1 --> Car
    (200, 0, 0): 12,  # Car 2 --> 19
    (150, 0, 0): 12,  # Car 3 --> Car
    (128, 0, 0): 12,  # Car 4 --> Car
    (182, 89, 6): 16,  # Bicycle 1 --> Bicycle
    (150, 50, 4): 16,  # Bicycle 2 --> Bicycle
    (90, 30, 1): 16,  # Bicycle 3 --> Bicycle
    (90, 30, 30): 16,  # Bicycle 4 --> Bicycle
    (204, 153, 255): 10,  # Pedestrian 1 --> Person
    (189, 73, 155): 10,  # Pedestrian 2 --> Person
    (239, 89, 191): 10,  # Pedestrian 3 --> Person
    (255, 128, 0): 13,  # Truck 1 --> Truck
    (200, 128, 0): 13,  # Truck 2 --> Truck
    (150, 128, 0): 13,  # Truck 3 --> Truck
    (0, 0, 100): 14,  # Tractor --> Utility vehicle (*not in CS)
    (0, 255, 0): 15,  # Small vehicles 1 --> Motorcycle
    (0, 200, 0): 15,  # Small vehicles 2 --> Motorcycle
    (0, 150, 0): 15,  # Small vehicles 3 --> Motorcycle
    (0, 128, 255): 6,  # Traffic signal 1 --> Traffic light
    (30, 28, 158): 6,  # Traffic signal 2 --> Traffic light
    (60, 28, 100): 6,  # Traffic signal 3 --> Traffic light
    (0, 255, 255): 7,  # Traffic sign 1 --> Traffic sign
    (30, 220, 220): 7,  # Traffic sign 2 --> Traffic sign
    (60, 157, 199): 7,  # Traffic sign 3 --> Traffic sign
    (255, 255, 0): 14,  # Utility vehicle 1 --> Utility vehicle (*not in CS)
    (255, 255, 200): 14,  # Utility vehicle 2 --> Utility vehicle (*not in CS)
    (233, 100, 0): 5,  # Sidebars --> Poles
    (110, 110, 0): 0,  # Speed bumper --> Road
    (128, 128, 0): 1,  # Curbstone --> Sidewalk
    (255, 193, 37): 0,  # Solid line --> Road
    (64, 0, 64): 17,  # Irrelevant signs --> Background (*not in CS)
    (185, 122, 87): 3,  # Road blocks --> Wall
    (139, 99, 108): 17,  # Non-drivable street --> Background (*not in CS)
    (210, 50, 115): 0,  # Zebra crossing --> Road
    (255, 0, 128): 17,  # Obstacles / trash --> Background (*not in CS)
    (255, 246, 143): 5,  # Poles --> Poles
    (150, 0, 150): 0,  # RD restricted area --> Road
    (204, 255, 153): 11,  # Animals --> Animal (*not in CS)
    (238, 162, 173): 4,  # Grid structure --> Fence
    (33, 44, 177): 6,  # Signal corpus --> Traffic light
    (180, 50, 180): 0,  # Drivable cobblestone --> Road
    (255, 70, 185): 17,  # Electronic traffic --> Background (*not in CS)
    (238, 233, 191): 0,  # Slow drive area --> Road
    (147, 253, 194): 8,  # Nature object --> Vegetation
    (150, 150, 200): 0,  # Parking area --> Road
    (180, 150, 200): 1,  # Sidewalk --> Sidewalk
    (72, 209, 204): 255,  # Ego car --> Static (void)
    (200, 125, 210): 0,  # Painted driv. instr. --> Road
    (159, 121, 238): 3,  # Traffic guide obj. --> Wall
    (128, 0, 255): 0,  # Dashed line --> Road
    (255, 0, 255): 0,  # RD normal street --> Road
    (135, 206, 255): 9,  # Sky --> Sky
    (241, 230, 255): 2,  # Buildings --> Building
    (96, 69, 143): 255,  # Blurred area --> Static (void)
    (53, 46, 82): 255,  # Rain dirt --> Dynamic (void)
}


def modify_label_filename(label_filepath, label_choice):
    """Returns a mmsegmentation-combatible label filename."""
    # Ensure that label filenames are modified only once
    if 'TrainIds.png' in label_filepath:
        return label_filepath

    label_filepath = label_filepath.replace('_label_', '_camera_')
    if label_choice == 'a2d2':
        label_filepath = label_filepath.replace('.png', LABEL_SUFFIX_34_CLS)
    elif label_choice == 'cityscapes':
        label_filepath = label_filepath.replace('.png', LABEL_SUFFIX_18_CLS)
    else:
        raise ValueError
    return label_filepath


def convert_a2d2_trainids(label_filepath, ignore_id=255):
    """Saves a new semantic label using the A2D2 label categories.

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

    seg_colors = list(SEG_COLOR_DICT_A2D2.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT_A2D2[seg_color]

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath, 'a2d2')
    label_img = mod_label.astype(np.uint8)
    mmcv.imwrite(label_img, label_filepath)


def convert_cityscapes_trainids(label_filepath, ignore_id=255):
    """Saves a new semantic label following the Cityscapes 'trainids' format.

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

    seg_colors = list(SEG_COLOR_DICT_CITYSCAPES.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT_CITYSCAPES[seg_color]

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath, 'cityscapes')
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
        # Partions string: [generic/path/to/file] [/] [filename]
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
                               val_ratio,
                               test_ratio,
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
        label_suffix: Label filename ending string.
        use_symlinks: Symbolically link existing files in the original A2D2
                      dataset directory. If false, files will be copied.
    """
    for r in [val_ratio, test_ratio]:
        assert r >= 0. and r < 1., 'Invalid ratio {}'.format(r)

    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'images'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'images', 'train'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'images', 'val'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'images', 'test'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'annotations', 'train'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'annotations', 'val'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'annotations', 'test'))

    # Lists containing all images and labels to symlinked
    img_filepaths = sorted(glob.glob(osp.join(a2d2_path, '*/camera/*/*.png')))

    if label_choice == 'a2d2':
        label_suffix = LABEL_SUFFIX_34_CLS
    elif label_choice == 'cityscapes':
        label_suffix = LABEL_SUFFIX_18_CLS
    else:
        raise ValueError
    ann_filepaths = sorted(
        glob.glob(osp.join(a2d2_path, '*/label/*/*{}'.format(label_suffix))))

    # Randomize order of (image, label) pairs
    pairs = list(zip(img_filepaths, ann_filepaths))
    random.shuffle(pairs)
    img_filepaths, ann_filepaths = zip(*pairs)

    # Split data according to given ratios
    total_samples = len(img_filepaths)
    train_ratio = 1.0 - val_ratio - test_ratio

    train_idx_end = int(np.ceil(train_ratio * (total_samples - 1)))
    val_idx_end = train_idx_end + int(np.ceil(val_ratio * total_samples))

    # Train split
    if train_on_val_and_test:
        train_img_paths = img_filepaths
        train_ann_paths = ann_filepaths
    else:
        train_img_paths = img_filepaths[:train_idx_end]
        train_ann_paths = ann_filepaths[:train_idx_end]
    # Val split
    val_img_paths = img_filepaths[train_idx_end:val_idx_end]
    val_ann_paths = ann_filepaths[train_idx_end:val_idx_end]
    # Test split
    test_img_paths = img_filepaths[val_idx_end:]
    test_ann_paths = ann_filepaths[val_idx_end:]

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
        default='cityscapes',
        help='Label conversion type choice: \'cityscapes\' (18 classes) or '
        '\'a2d2\' (34 classes)')
    parser.add_argument(
        '--val', default=0.103, type=float, help='Validation set sample ratio')
    parser.add_argument(
        '--test', default=0.197, type=float, help='Test set sample ratio')
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

        The function 'convert_TYPE_trainids()' converts all instance
        segmentation to their corresponding categorical segmentation and saves
        them as new label image files.

        Label choice 'cityscapes' (default) results in labels with 18 classes
        with the filename suffix '_18LabelTrainIds.png'.

        Label choice 'a2d2' results in labels with 34 classes with the filename
        suffix '_34LabelTrainIds.png'.

    The default arguments result in merging of the original 38 semantic classes
    into a Cityscapes-like 18 class label setup. The official A2D2 paper
    presents benchmark results in an unspecified but presumptively similar
    class taxonomy. (ref: p.8 "4. Experiment: Semantic segmentation").

    Add `--choice a2d2` to use the original 34 A2D2 semantic classes.

    Samples are randomly split into 'train', 'val' and 'test' sets, each
    consisting of 28,894 samples (70.0%), 4,252 samples (10.3%) and 8,131
    samples (19.7%), respectively.

    Add the optional argument `--train-on-val-and-test` to train on the entire
    dataset.

    Add `--nproc N` for multiprocessing using N threads.

    NOTE: The following segmentation classes are ignored (i.e. trainIds 255):
          - Ego car:  A calibrated system should a priori know what input
                      region corresponds to the ego vehicle.
          - Blurred area: Ambiguous semantic.
          - Rain dirt: Ambiguous semantic.

          The following segmentation class is merged due to extreme rarity:
          - Speed bumper --> RD normal street (randomly parsing 50% of dataset
            results in only one sample containing the 'speed_bumper' semantic)

    Directory restructuring:
        A2D2 files are not arranged in the required 'train/val/test' directory
        structure.

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
        if seg_choice == 'cityscapes':
            if args.nproc > 1:
                mmcv.track_parallel_progress(convert_cityscapes_trainids,
                                             label_filepaths, args.nproc)
            else:
                mmcv.track_progress(convert_cityscapes_trainids,
                                    label_filepaths)
        elif seg_choice == 'a2d2':
            if args.nproc > 1:
                mmcv.track_parallel_progress(convert_a2d2_trainids,
                                             label_filepaths, args.nproc)
            else:
                mmcv.track_progress(convert_a2d2_trainids, label_filepaths)
        else:
            raise ValueError

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    if args.restruct:
        restructure_a2d2_directory(out_dir, args.val, args.test, args.choice,
                                   args.train_on_val_and_test, args.symlink)


if __name__ == '__main__':
    main()
