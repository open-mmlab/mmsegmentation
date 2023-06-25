import glob
import os
import shutil

import numpy as np
from PIL import Image


def check_maskid(train_imgs):
    for i in train_masks:
        img = Image.open(i)
        print(np.unique(np.array(img)))


def reformulate_file(image_list, mask_list):
    file_list = []
    for idx, (imgp,
              maskp) in enumerate(zip(sorted(image_list), sorted(mask_list))):
        item = {'image': imgp, 'label': maskp}
        file_list.append(item)
    return file_list


def convert_maskid(mask):
    # add mask id conversion
    arr_mask = np.array(mask).astype(np.uint8)
    arr_mask[arr_mask == 255] = 1
    return Image.fromarray(arr_mask)


def check_file_exist(pair_list):
    rel_path = os.getcwd()
    for idx, sample in enumerate(pair_list):
        image_path = sample['image']
        assert os.path.exists(os.path.join(rel_path, image_path))
        if 'label' in sample:
            mask_path = sample['label']
            assert os.path.exists(os.path.join(rel_path, mask_path))
    print('all file path ok!')


def process_dataset(file_lists, part_dir_dict):
    for ith, part in enumerate(file_lists):
        part_dir = part_dir_dict[ith]
        for sample in part:
            # read image and mask
            image_path = sample['image']
            if 'label' in sample:
                mask_path = sample['label']

            basename = os.path.basename(image_path)
            targetname = basename.split('.')[0]  # from image name

            # check image file
            img_save_path = os.path.join(root_path, 'images', part_dir,
                                         targetname + save_img_suffix)
            if not os.path.exists(img_save_path):
                if not image_path.endswith('.png'):
                    src = Image.open(image_path)
                    src.save(img_save_path)
                else:
                    shutil.copy(image_path, img_save_path)

            if mask_path is not None:
                mask_save_path = os.path.join(root_path, 'masks', part_dir,
                                              targetname + save_seg_map_suffix)
                if not os.path.exists(mask_save_path):
                    # check mask file
                    mask = Image.open(mask_path).convert('L')
                    # convert mask id
                    mask = convert_maskid(mask)
                    if not mask_path.endswith('.png'):
                        mask.save(mask_save_path)
                    else:
                        mask.save(mask_save_path)

        # print image num
        part_dir_folder = os.path.join(root_path, 'images', part_dir)
        print(
            f'{part_dir} has {len(os.listdir(part_dir_folder))} images completed!'  # noqa
        )


if __name__ == '__main__':

    root_path = 'data/'  # original file
    img_suffix = '.jpg'
    seg_map_suffix = '.png'
    save_img_suffix = '.png'
    save_seg_map_suffix = '.png'

    train_imgs = glob.glob('data/ISIC-2017_Training_Data/*' + img_suffix)
    train_masks = glob.glob('data/ISIC-2017_Training_Part1_GroundTruth/*' +
                            seg_map_suffix)

    val_imgs = glob.glob('data/ISIC-2017_Validation_Data/*' + img_suffix)
    val_masks = glob.glob('data/ISIC-2017_Validation_Part1_GroundTruth/*' +
                          seg_map_suffix)

    test_imgs = glob.glob('data/ISIC-2017_Test_v2_Data/*' + img_suffix)
    test_masks = glob.glob('data/ISIC-2017_Test_v2_Part1_GroundTruth/*' +
                           seg_map_suffix)

    assert len(train_imgs) == len(train_masks)
    assert len(val_imgs) == len(val_masks)
    assert len(test_imgs) == len(test_masks)

    os.system('mkdir -p ' + root_path + 'images/train/')
    os.system('mkdir -p ' + root_path + 'images/val/')
    os.system('mkdir -p ' + root_path + 'images/test/')
    os.system('mkdir -p ' + root_path + 'masks/train/')
    os.system('mkdir -p ' + root_path + 'masks/val/')
    os.system('mkdir -p ' + root_path + 'masks/test/')

    part_dir_dict = {0: 'train/', 1: 'val/', 2: 'test/'}

    train_pair_list = reformulate_file(train_imgs, train_masks)
    val_pair_list = reformulate_file(val_imgs, val_masks)
    test_pair_list = reformulate_file(test_imgs, test_masks)

    check_file_exist(train_pair_list)
    check_file_exist(val_pair_list)
    check_file_exist(test_pair_list)

    part_dir_dict = {0: 'train/', 1: 'val/', 2: 'test/'}
    process_dataset([train_pair_list, val_pair_list, test_pair_list],
                    part_dir_dict)
