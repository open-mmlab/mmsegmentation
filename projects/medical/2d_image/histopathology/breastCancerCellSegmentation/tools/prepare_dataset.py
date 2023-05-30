import argparse
import glob
import os

from sklearn.model_selection import train_test_split


def save_anno(img_list, file_path, remove_suffix=True):
    if remove_suffix:  # （文件路径从data/${image/masks}之后的相对路径开始）
        img_list = [
            '/'.join(img_path.split('/')[-2:]) for img_path in img_list
        ]  # noqa
        img_list = [
            '.'.join(img_path.split('.')[:-1]) for img_path in img_list
        ]  # noqa
    with open(file_path, 'w') as file_:
        for x in list(img_list):
            file_.write(x + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', default='data/')
    args = parser.parse_args()
    data_root = args.data_root

    # 1. 划分训练集、验证集
    # 1.1 获取所有图片路径
    img_list = glob.glob(os.path.join(data_root, 'images', '*.tif'))
    img_list.sort()
    mask_list = glob.glob(os.path.join(data_root, 'masks', '*.TIF'))
    mask_list.sort()
    assert len(img_list) == len(mask_list)
    # 1.2 划分训练集、验证集、测试集
    train_img_list, val_img_list, train_mask_list, val_mask_list = train_test_split(  # noqa
        img_list, mask_list, test_size=0.2, random_state=42)
    # 1.3 保存划分结果
    save_anno(train_img_list, os.path.join(data_root, 'train.txt'))
    save_anno(val_img_list, os.path.join(data_root, 'val.txt'))
