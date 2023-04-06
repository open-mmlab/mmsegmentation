# import os
# import shutil
# from tqdm import tqdm

# rp = './data/SegPC2021/SegPC2021/TCIA_SegPC_dataset/TCIA_SegPC_dataset/TCIA_SegPC_dataset/validation/validation/y'  # noqa

# for im in tqdm(os.listdir(rp)):
#     im_p = os.path.join(rp, im)
#     shutil.copy(im_p, './data/SegPC2021/masks/val/'+im)

#     im_im = im.split('_')[0]+'.bmp'
#     im_im_p = os.path.join('./data/SegPC2021/SegPC2021/TCIA_SegPC_dataset/TCIA_SegPC_dataset/TCIA_SegPC_dataset/validation/validation/x', im_im)  # noqa
#     shutil.copy(im_im_p, './data/SegPC2021/images/val/'+im)

import argparse
import os

# import glob


def save_anno(img_list, file_path, remove_suffix=True):
    if remove_suffix:  # （文件路径从data/${image/masks}之后的相对路径开始）
        img_list = [img_path.split('/images/')[-1] for img_path in img_list]
        img_list = [
            '.'.join(img_path.split('.')[:-1]) for img_path in img_list
        ]  # noqa
    with open(file_path, 'w') as file_:
        for x in list(img_list):
            file_.write(x + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    parser.add_argument('suffix')
    args = parser.parse_args()
    data_root = args.data_root
    suffix = args.suffix
    # if os.path.exists(os.path.join(data_root, "images/val")):
    #     x_val = sorted(glob.glob(os.path.join(data_root, "images/val/", "*." + suffix)))  # noqa
    #     save_anno(x_val, data_root + '/val.txt')
    # if os.path.exists(os.path.join(data_root, "images/test")):
    #     x_test = sorted(glob.glob(os.path.join(data_root, "images/test/", "*." + suffix)))  # noqa
    #     save_anno(x_test, data_root + '/test.txt')
    # if os.path.exists(os.path.join(data_root, "images/train")):
    #     x_train= sorted(glob.glob(os.path.join(data_root, "images/train/", "*." + suffix)))  # noqa
    #     save_anno(x_train, data_root + '/train.txt')
    # ----------生成md5值以及包含无标签image的list，pr时该部分代码将被删除------------
    import hashlib
    all_imgs = []
    train_imgs = []
    val_imgs = []
    test_imgs = []
    for fpath, dirname, fnames in os.walk(os.path.join(data_root, 'images')):
        for fname in fnames:
            if fname.endswith('.' + suffix):
                img_path = os.path.join(fpath, fname)
                all_imgs.append(img_path)

                if img_path.split('/images/')[-1].startswith('train'):
                    train_imgs.append(os.path.join(fpath, fname))
                elif img_path.split('/images/')[-1].startswith('val'):
                    val_imgs.append(os.path.join(fpath, fname))
                elif img_path.split('/images/')[-1].startswith('test'):
                    test_imgs.append(os.path.join(fpath, fname))
    f_ = open(data_root + '/images_md5_list.txt', 'w')
    for img in sorted(all_imgs):
        save_name = img.split('/images/')[-1]
        with open(img, 'rb') as fd:
            fmd5 = hashlib.md5(fd.read()).hexdigest()
        f_.write(fmd5 + '\t' + save_name + '\n')

    f_.close()
    if len(val_imgs) > 0:
        save_anno(val_imgs, data_root + '/val.txt')
    if len(test_imgs) > 0:
        save_anno(test_imgs, data_root + '/test.txt')
    if len(train_imgs) > 0:
        save_anno(train_imgs, data_root + '/train.txt')
