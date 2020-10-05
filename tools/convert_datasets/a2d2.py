import argparse
import os.path as osp
from os import symlink
import glob
import random
from PIL import Image
import numpy as np

import mmcv

# key: RGB color, value: trainId (from Cityscapes)
SEG_COLOR_DICT = {
    (255, 0, 0):      13,  # Car 1 --> Car
    (200, 0, 0):      13,  # Car 2 --> Car
    (150, 0, 0):      13,  # Car 3 --> Car
    (128, 0, 0):      13,  # Car 4 --> Car
    (182, 89, 6):     18,  # Bicycle 1 --> Bicycle
    (150, 50, 4):     18,  # Bicycle 2 --> Bicycle
    (90, 30, 1):      18,  # Bicycle 3 --> Bicycle
    (90, 30, 30):     18,  # Bicycle 4 --> Bicycle
    (204, 153, 255):  11,  # Pedestrian 1 --> Person
    (189, 73, 155):   11,  # Pedestrian 2 --> Person
    (239, 89, 191):   11,  # Pedestrian 3 --> Person
    (255, 128, 0):    14,  # Truck 1 --> Truck
    (200, 128, 0):    14,  # Truck 2 --> Truck
    (150, 128, 0):    14,  # Truck 3 --> Truck
    (0, 255, 0):      17,  # Small vehicles 1 --> Motorcycle ?
    (0, 200, 0):      17,  # Small vehicles 2 --> Motorcycle ?
    (0, 150, 0):      17,  # Small vehicles 3 --> Motorcycle ?
    (0, 128, 255):     6,  # Traffic signal 1 --> Traffic light
    (30, 28, 158):     6,  # Traffic signal 2 --> Traffic light
    (60, 28, 100):     6,  # Traffic signal 3 --> Traffic light
    (0, 255, 255):     7,  # Traffic sign 1 --> Traffic sign
    (30, 220, 220):    7,  # Traffic sign 2 --> Traffic sign
    (60, 157, 199):    7,  # Traffic sign 3 --> Traffic sign
    (255, 255, 0):   255,  # Utility vehicle 1 --> Ignore ?
    (255, 255, 200): 255,  # Utility vehicle 2 --> Ignore ?
    (233, 100, 0):    25,  # Sidebars --> Ignore ?
    (110, 110, 0):     0,  # Speed bumper --> Road
    (128, 128, 0):     1,  # Curbstone --> Sidewalk
    (255, 193, 37):    0,  # Solid line --> Road
    (64, 0, 64):     255,  # Irrelevant signs --> Ignore
    (185, 122, 87):    4,  # Road blocks --> Fence ?
    (0, 0, 100):     255,  # Tractor --> Ignore ?
    (139, 99, 108):    0,  # Non-drivable street --> Road
    (210, 50, 115):    0,  # Zebra crossing --> Road
    (255, 0, 128):   255,  # Obstacles / trash --> Ignore ?
    (255, 246, 143):   5,  # Poles --> Poles
    (150, 0, 150):     0,  # RD restricted area --> Road ?
    (204, 255, 153): 255,  # Animals --> Ignore ?
    (238, 162, 173):   0,  # Grid structure --> Road ???
    (33, 44, 177):     6,  # Signal corpus --> Traffic light ?
    (180, 50, 180):    0,  # Drivable cobblestone --> Road
    (255, 70, 185):  255,  # Electronic traffic --> Ignore ???
    (238, 233, 191):   0,  # Slow drive area --> Road
    (147, 253, 194):   8,  # Nature object --> Vegetation
    (150, 150, 200):   0,  # Parking area --> Road
    (180, 150, 200):   1,  # Sidewalk --> Sidewalk 
    (72, 209, 204):  255,  # Ego car --> Ignore
    (200, 125, 210):   0,  # Painted driv. instr. --> Road ???
    (159, 121, 238):   0,  # Traffic guide obj. --> Road ???
    (128, 0, 255):     0,  # Dashed line --> Road
    (255, 0, 255):     0,  # RD normal street --> Road
    (135, 206, 255):  10,  # Sky --> Sky
    (241, 230, 255):   2,  # Buildings --> Building
    (96, 69, 143):   255,  # Blurred area --> Ignore ?
    (53, 46, 82):    255,  # Rain dirt --> Ignore ?
}


def modify_label_filename(label_filepath):
    ''' Returns a mmsegmentation-combatible label filename.
    '''
    label_filepath = label_filepath.replace('_label_', '_camera_')
    label_filepath = label_filepath.replace('.png', '_labelTrainIds.png')
    return label_filepath


def convert_label_values_to_trainids(label_filepath, ignore_id=255):
    '''Saves a new semantic label following the Cityscapes 'trainids' format.

    The new image is saved into the same directory as the original image having
    an additional suffix.

    Args:
        label_filepath: Path to the original semantic label.
        ignore_id: Default value for unlabeled elements.
    '''
    # Read label file as Numpy array (H, W, 3)
    orig_label = Image.open(label_filepath)
    orig_label = np.array(orig_label, dtype=np.int)

    # Empty array with all elements set as 'ignore id' label
    H, W, _ = orig_label.shape
    mod_label = ignore_id*np.ones((H,W), dtype=np.int)

    seg_colors = list(SEG_COLOR_DICT.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(np.bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT[seg_color]

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath)
    label_img = Image.fromarray(mod_label.astype(np.uint8))
    label_img.save(label_filepath)


def restructure_a2d2_directory(a2d2_path, val_ratio, test_ratio, 
                               label_suffix='_labelTrainIds.png'):
    '''Creates a new directory structure and symlink existing files into it.

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
        a2d2_path:
        val_ratio:
        test_ratio:
        label_suffix:
    '''

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
    img_filepaths = sorted(glob.glob(osp.join(a2d2_path, 
                           '*/camera/cam_front_center/*.png')))
    ann_filepaths = sorted(glob.glob(osp.join(a2d2_path,
                           f'*/label/cam_front_center/*{label_suffix}')))

    # Split data according to given ratios
    total_samples = len(img_filepaths)
    train_ratio = 1.0 - val_ratio - test_ratio

    # For writing the train/val/test samples sammaries to file
    train_filenames = []
    val_filenames = []
    test_filenames = []

    # Create symlinks file-by-file
    for sample_idx in range(total_samples):

        img_filepath = img_filepaths[sample_idx]
        ann_filepath = ann_filepaths[sample_idx]

        # Partions string: [generic/path/to/file] [/] [filename]
        img_filename = img_filepath.rpartition('/')[2]
        ann_filename = ann_filepath.rpartition('/')[2]

        train_idx_end = int(train_ratio*total_samples)
        val_idx_end = train_idx_end + int(val_ratio*total_samples)

        if sample_idx < train_idx_end:
            split = 'train'
            train_filenames.append(img_filename[:-4])
        elif sample_idx < val_idx_end:
            split = 'val'
            val_filenames.append(img_filename[:-4])
        else:
            split = 'test'
            test_filenames.append(img_filename[:-4])

        img_symlink_path = osp.join(a2d2_path, 'img_dir', split, img_filename)
        ann_symlink_path = osp.join(a2d2_path, 'ann_dir', split, ann_filename)

        # NOTE: Can only create new symlinks if no priors ones exists
        try:
            symlink(img_filepath, img_symlink_path)
        except FileExistsError:
            pass
        try:
            symlink(ann_filepath, ann_symlink_path)
        except FileExistsError:
            pass

    # Sample summary files
    with open(osp.join(a2d2_path, f'train.txt'), 'w') as file:
        file.writelines(fn + '\n' for fn in train_filenames)
    with open(osp.join(a2d2_path, f'val.txt'), 'w') as file:
        file.writelines(fn + '\n' for fn in val_filenames)
    with open(osp.join(a2d2_path, f'test.txt'), 'w') as file:
        file.writelines(fn + '\n' for fn in test_filenames)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert A2D2 annotations to TrainIds')
    parser.add_argument('a2d2_path', help='A2D2 segmentation data path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('--conv', default=True, help='Convert label images')
    parser.add_argument(
        '--restruct', default=True, help="Restructure directory structure")
    parser.add_argument(
        '--val', default=0.1, type=float, help="Validation set sample ratio")
    parser.add_argument(
        '--test', default=0.1, type=float, help="Test set sample ratio")
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    '''
    '''
    args = parse_args()
    a2d2_path = args.a2d2_path
    out_dir = args.out_dir if args.out_dir else a2d2_path
    mmcv.mkdir_or_exist(out_dir)

    # Create a list of filepaths to all original labels
    # NOTE: Original label files have a number before '.png'
    label_filepaths = glob.glob(osp.join(a2d2_path, '*/label/*/*[0-9].png'))

    # Convert segmentation images to the Cityscapes 'TrainIds' values
    if args.conv == True:
        if args.nproc > 1:
            mmcv.track_parallel_progress(convert_label_values_to_trainids, label_filepaths, args.nproc)
        else:
            mmcv.track_progress(convert_label_values_to_trainids, label_filepaths)

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    if args.restruct:
        restructure_a2d2_directory(out_dir, args.val, args.test)




if __name__ == '__main__':
    main()