# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument(
        '--imgfile_prefix',
        type=str,
        required=True,
        help='The prefix of output image file')
    parser.add_argument(
        '--method',
        default='example submission',
        help='Method name in method description file(method.txt).')
    parser.add_argument(
        '--arch',
        metavar='ARCH',
        help='The model architecture in method description file(method.txt).')
    parser.add_argument(
        '--train_data',
        default='null',
        help='Training data in method description file(method.txt).')
    parser.add_argument(
        '--train_scheme',
        default='null',
        help='Training scheme in method description file(method.txt), '
        'e.g., SSL, Sup, SSL+Sup.')
    parser.add_argument(
        '--link',
        default='null',
        help='Paper/project link in method description file(method.txt).')
    parser.add_argument(
        '--description',
        default='null',
        help='Method description in method description file(method.txt).')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    method = 'Method name: {}\n'.format(args.method) + \
        'Training data: {}\nTraining scheme: {}\n'.format(
            args.train_data, args.train_scheme) + \
        'Networks: {}\nPaper/Project link: {}\n'.format(
            args.arch, args.link) + \
        'Method description: {}'.format(args.description)
    with open(os.path.join(args.imgfile_prefix, 'method.txt'), 'w') as f:
        f.write(method)

    # zip for submission
    shutil.make_archive(
        args.imgfile_prefix, 'zip', root_dir=args.imgfile_prefix)
