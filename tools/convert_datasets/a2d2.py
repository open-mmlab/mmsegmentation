import argparse
import glob
import os.path as osp
import random
from os import symlink
from shutil import copyfile

import mmcv
import numpy as np

# Dictionaries specifying which A2D2 segmentation color corresponds to

# A2D2 'trainId' value
#   key: RGB color, value: trainId
SEG_COLOR_DICT_A2D2 = {
    (255, 0, 0): 28,  # Car 1
    (200, 0, 0): 28,  # Car 2
    (150, 0, 0): 28,  # Car 3
    (128, 0, 0): 28,  # Car 4
    (182, 89, 6): 27,  # Bicycle 1
    (150, 50, 4): 27,  # Bicycle 2
    (90, 30, 1): 27,  # Bicycle 3
    (90, 30, 30): 27,  # Bicycle 4
    (204, 153, 255): 26,  # Pedestrian 1
    (189, 73, 155): 26,  # Pedestrian 2
    (239, 89, 191): 26,  # Pedestrian 3
    (255, 128, 0): 30,  # Truck 1
    (200, 128, 0): 30,  # Truck 2
    (150, 128, 0): 30,  # Truck 3
    (0, 255, 0): 32,  # Small vehicles 1
    (0, 200, 0): 32,  # Small vehicles 2
    (0, 150, 0): 32,  # Small vehicles 3
    (0, 128, 255): 19,  # Traffic signal 1
    (30, 28, 158): 19,  # Traffic signal 2
    (60, 28, 100): 19,  # Traffic signal 3
    (0, 255, 255): 20,  # Traffic sign 1
    (30, 220, 220): 20,  # Traffic sign 2
    (60, 157, 199): 20,  # Traffic sign 3
    (255, 255, 0): 29,  # Utility vehicle 1
    (255, 255, 200): 29,  # Utility vehicle 2
    (233, 100, 0): 16,  # Sidebars
    (110, 110, 0): 12,  # Speed bumper
    (128, 128, 0): 14,  # Curbstone
    (255, 193, 37): 6,  # Solid line
    (64, 0, 64): 22,  # Irrelevant signs
    (185, 122, 87): 17,  # Road blocks
    (0, 0, 100): 31,  # Tractor
    (139, 99, 108): 1,  # Non-drivable street
    (210, 50, 115): 8,  # Zebra crossing
    (255, 0, 128): 34,  # Obstacles / trash
    (255, 246, 143): 18,  # Poles
    (150, 0, 150): 2,  # RD restricted area
    (204, 255, 153): 33,  # Animals
    (238, 162, 173): 9,  # Grid structure
    (33, 44, 177): 21,  # Signal corpus
    (180, 50, 180): 3,  # Drivable cobblestone
    (255, 70, 185): 23,  # Electronic traffic
    (238, 233, 191): 4,  # Slow drive area
    (147, 253, 194): 24,  # Nature object
    (150, 150, 200): 5,  # Parking area
    (180, 150, 200): 13,  # Sidewalk
    (72, 209, 204): 255,  # Ego car
    (200, 125, 210): 11,  # Painted driv. instr.
    (159, 121, 238): 10,  # Traffic guide obj.
    (128, 0, 255): 7,  # Dashed line
    (255, 0, 255): 0,  # RD normal street
    (135, 206, 255): 25,  # Sky
    (241, 230, 255): 15,  # Buildings
    (96, 69, 143): 255,  # Blurred area
    (53, 46, 82): 255,  # Rain dirt
}

# Cityscapes 'trainId' value
#   key: RGB color, value: trainId (from Cityscapes)
SEG_COLOR_DICT_CITYSCAPES = {
    (255, 0, 0): 13,  # Car 1 --> Car
    (200, 0, 0): 13,  # Car 2 --> Car
    (150, 0, 0): 13,  # Car 3 --> Car
    (128, 0, 0): 13,  # Car 4 --> Car
    (182, 89, 6): 18,  # Bicycle 1 --> Bicycle
    (150, 50, 4): 18,  # Bicycle 2 --> Bicycle
    (90, 30, 1): 18,  # Bicycle 3 --> Bicycle
    (90, 30, 30): 18,  # Bicycle 4 --> Bicycle
    (204, 153, 255): 11,  # Pedestrian 1 --> Person
    (189, 73, 155): 11,  # Pedestrian 2 --> Person
    (239, 89, 191): 11,  # Pedestrian 3 --> Person
    (255, 128, 0): 14,  # Truck 1 --> Truck
    (200, 128, 0): 14,  # Truck 2 --> Truck
    (150, 128, 0): 14,  # Truck 3 --> Truck
    (0, 0, 100): 14,  # Tractor --> Truck ?
    (0, 255, 0): 17,  # Small vehicles 1 --> Motorcycle ?
    (0, 200, 0): 17,  # Small vehicles 2 --> Motorcycle ?
    (0, 150, 0): 17,  # Small vehicles 3 --> Motorcycle ?
    (0, 128, 255): 6,  # Traffic signal 1 --> Traffic light
    (30, 28, 158): 6,  # Traffic signal 2 --> Traffic light
    (60, 28, 100): 6,  # Traffic signal 3 --> Traffic light
    (0, 255, 255): 7,  # Traffic sign 1 --> Traffic sign
    (30, 220, 220): 7,  # Traffic sign 2 --> Traffic sign
    (60, 157, 199): 7,  # Traffic sign 3 --> Traffic sign
    (255, 255, 0): 13,  # Utility vehicle 1 --> Car ?
    (255, 255, 200): 13,  # Utility vehicle 2 --> Car ?
    (233, 100, 0): 4,  # Sidebars --> Fence
    (110, 110, 0): 0,  # Speed bumper --> Road
    (128, 128, 0): 1,  # Curbstone --> Sidewalk
    (255, 193, 37): 0,  # Solid line --> Road
    (64, 0, 64): 255,  # Irrelevant signs --> Ignore
    (185, 122, 87): 4,  # Road blocks --> Fence
    (139, 99, 108): 0,  # Non-drivable street --> Road
    (210, 50, 115): 0,  # Zebra crossing --> Road
    (255, 0, 128): 255,  # Obstacles / trash --> Ignore ?
    (255, 246, 143): 5,  # Poles --> Poles
    (150, 0, 150): 0,  # RD restricted area --> Road ?
    (204, 255, 153): 255,  # Animals --> Ignore ?
    (238, 162, 173): 0,  # Grid structure --> Road ???
    (33, 44, 177): 6,  # Signal corpus --> Traffic light ?
    (180, 50, 180): 0,  # Drivable cobblestone --> Road
    (255, 70, 185): 255,  # Electronic traffic --> Ignore ???
    (238, 233, 191): 0,  # Slow drive area --> Road
    (147, 253, 194): 8,  # Nature object --> Vegetation
    (150, 150, 200): 0,  # Parking area --> Road
    (180, 150, 200): 1,  # Sidewalk --> Sidewalk
    (72, 209, 204): 255,  # Ego car --> Ignore
    (200, 125, 210): 0,  # Painted driv. instr. --> Road ???
    (159, 121, 238): 0,  # Traffic guide obj. --> Road ???
    (128, 0, 255): 0,  # Dashed line --> Road
    (255, 0, 255): 0,  # RD normal street --> Road
    (135, 206, 255): 10,  # Sky --> Sky
    (241, 230, 255): 2,  # Buildings --> Building
    (96, 69, 143): 255,  # Blurred area --> Ignore ?
    (53, 46, 82): 255,  # Rain dirt --> Ignore ?
}


def modify_label_filename(label_filepath):
    """Returns a mmsegmentation-combatible label filename."""
    label_filepath = label_filepath.replace('_label_', '_camera_')
    label_filepath = label_filepath.replace('.png', '_labelTrainIds.png')
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
    mod_label = ignore_id * np.ones((H, W), dtype=np.int)

    seg_colors = list(SEG_COLOR_DICT_A2D2.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(np.bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT_A2D2[seg_color]

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath)
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
    mod_label = ignore_id * np.ones((H, W), dtype=np.int)

    seg_colors = list(SEG_COLOR_DICT_CITYSCAPES.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(np.bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT_CITYSCAPES[seg_color]

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath)
    label_img = mod_label.astype(np.uint8)
    mmcv.imwrite(label_img, label_filepath)


def restructure_a2d2_directory(a2d2_path,
                               val_ratio,
                               test_ratio,
                               use_symlinks=True,
                               label_suffix='_labelTrainIds.png'):
    """Creates a new directory structure and link existing files into it.

    Required to make the A2D2 dataset conform to the mmsegmentation frameworks
    expected dataset structure.

    my_dataset
    └── img_dir
    │   ├── train
    │   │   ├── xxx{img_suffix}
    |   |   ...
    │   ├── val
    │   │   ├── yyy{img_suffix}
    │   │   ...
    │   ...
    └── ann_dir
        ├── train
        │   ├── xxx{seg_map_suffix}
        |   ...
        ├── val
        |   ├── yyy{seg_map_suffix}
        ... ...

    Args:
        a2d2_path: Absolute path to the A2D2 'camera_lidar_semantic' directory.
        val_ratio: Float value representing ratio of validation samples.
        test_ratio: Float value representing ratio of test samples.
        label_suffix: Label filename ending string.
        use_symlinks: Symbolically link existing files in the original A2D2
                      dataset directory. If false, files will be copied.
    """

    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'img_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(a2d2_path, 'ann_dir', 'test'))

    # Lists containing all images and labels to symlinked
    img_filepaths = sorted(glob.glob(osp.join(a2d2_path, '*/camera/*/*.png')))
    ann_filepaths = sorted(
        glob.glob(osp.join(a2d2_path, '*/label/*/*{}'.format(label_suffix))))

    # Randomize order of (image, label) pairs
    pairs = list(zip(img_filepaths, ann_filepaths))
    random.shuffle(pairs)
    img_filepaths, ann_filepaths = zip(*pairs)

    # Split data according to given ratios
    total_samples = len(img_filepaths)
    train_ratio = 1.0 - val_ratio - test_ratio

    train_idx_end = int(np.floor(train_ratio * (total_samples - 1)))
    val_idx_end = train_idx_end + int(np.ceil(val_ratio * total_samples))

    # Create symlinks file-by-file
    for sample_idx in range(total_samples):

        img_filepath = img_filepaths[sample_idx]
        ann_filepath = ann_filepaths[sample_idx]

        # Partions string: [generic/path/to/file] [/] [filename]
        img_filename = img_filepath.rpartition('/')[2]
        ann_filename = ann_filepath.rpartition('/')[2]

        if sample_idx <= train_idx_end:
            split = 'train'
        elif sample_idx <= val_idx_end:
            split = 'val'
        else:
            split = 'test'

        img_link_path = osp.join(a2d2_path, 'img_dir', split, img_filename)
        ann_link_path = osp.join(a2d2_path, 'ann_dir', split, ann_filename)

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
        '--choice', default='cityscapes', help='Label conversion type choice')
    parser.add_argument(
        '--val', default=0.02, type=float, help='Validation set sample ratio')
    parser.add_argument(
        '--test', default=0., type=float, help='Test set sample ratio')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    parser.add_argument(
        '--symlink', default=True, help='Use symbolic links insted of copies')
    args = parser.parse_args()
    return args


def main():
    """Program for making Audi's A2D2 dataset compatible with mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/a2d2)

    Segmentation label conversion:
        The A2D2 labels are instance segmentations (i.e. car_1, car_2, ...),
        while semantic segmentation requires categorical segmentations.

        The function 'convert_TYPE_trainids()' converts all instance
        segmentation to their corresponding categorical segmentation and saves
        them as new label image files.

        Conversion type options
            A2D2: Generates segmentations using inherent categories.
            Cityscapes: Generates segmentations according to the categories and
                        indexing (i.e. 'trainIds') as in Cityscapes.

    Directory restructuring:
        A2D2 files are not arranged in the required 'train/val/test' directory
        structure.

        The function 'restructure_a2d2_directory' creates a new compatible
        directory structure in the root directory, and fills it with symbolic
        links or file copies to the input and segmentation label images.

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
        restructure_a2d2_directory(out_dir, args.val, args.test, args.symlink)


if __name__ == '__main__':
    main()
