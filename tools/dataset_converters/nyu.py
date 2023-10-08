# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import shutil
import tempfile
import zipfile

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert NYU Depth dataset to mmsegmentation format')
    parser.add_argument('raw_data', help='the path of raw data')
    parser.add_argument(
        '-o', '--out_dir', help='output path', default='./data/nyu')
    args = parser.parse_args()
    return args


def reorganize(raw_data_dir: str, out_dir: str):
    """Reorganize NYU Depth dataset files into the required directory
    structure.

    Args:
        raw_data_dir (str): Path to the raw data directory.
        out_dir (str): Output directory for the organized dataset.
    """

    def move_data(data_list, dst_prefix, fname_func):
        """Move data files from source to destination directory.

        Args:
            data_list (list): List of data file paths.
            dst_prefix (str): Prefix to be added to destination paths.
            fname_func (callable): Function to process file names
        """
        for data_item in data_list:
            data_item = data_item.strip().strip('/')
            new_item = fname_func(data_item)
            shutil.move(
                osp.join(raw_data_dir, data_item),
                osp.join(out_dir, dst_prefix, new_item))

    def process_phase(phase):
        """Process a dataset phase (e.g., 'train' or 'test')."""
        with open(osp.join(raw_data_dir, f'nyu_{phase}.txt')) as f:
            data = filter(lambda x: len(x.strip()) > 0, f.readlines())
            data = map(lambda x: x.split()[:2], data)
            images, annos = zip(*data)

            move_data(images, f'images/{phase}',
                      lambda x: x.replace('/rgb', ''))
            move_data(annos, f'annotations/{phase}',
                      lambda x: x.replace('/sync_depth', ''))

    process_phase('train')
    process_phase('test')


def main():
    args = parse_args()

    print('Making directories...')
    mkdir_or_exist(args.out_dir)
    for subdir in [
            'images/train', 'images/test', 'annotations/train',
            'annotations/test'
    ]:
        mkdir_or_exist(osp.join(args.out_dir, subdir))

    print('Generating images and annotations...')

    if args.raw_data.endswith('.zip'):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_file = zipfile.ZipFile(args.raw_data)
            zip_file.extractall(tmp_dir)
            reorganize(osp.join(tmp_dir, 'nyu'), args.out_dir)
    else:
        assert osp.isdir(
            args.raw_data
        ), 'the argument --raw-data should be either a zip file or directory.'
        reorganize(args.raw_data, args.out_dir)

    print('Done!')


if __name__ == '__main__':
    main()
