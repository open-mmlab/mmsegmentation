#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import os
import os.path as osp
import re
import sys

HEADER = 'Copyright (c) OpenMMLab. All rights reserved.\n'
HEADER_KEYWORDS = {'Copyright', 'License'}


def contains_header(lines, comment_symbol, max_header_lines):
    for line in lines[:max_header_lines]:
        if line.startswith('#!'):
            # skip shebang line
            continue
        elif re.match(f'{comment_symbol}.*({"|".join(HEADER_KEYWORDS)})',
                      line):
            return True

    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'files',
        type=str,
        nargs='*',
        help='Files to add copyright header. If an empty list is given, '
        'search target files according to "--src", "--exclude" and '
        '"--suffixes"')
    parser.add_argument(
        '--src', type=str, default=None, help='Root path to search files.')
    parser.add_argument(
        '--exclude', type=str, default=None, help='Path to exclude in search.')
    parser.add_argument(
        '--suffixes',
        type=str,
        nargs='+',
        default=['.py', '.c', '.cpp', '.cu', '.sh'],
        help='Only files with one of the given suffixes will be searched.')
    parser.add_argument(
        '--max-header-lines',
        type=int,
        default=5,
        help='Only checkout copyright information in the first several lines '
        'of a file.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    file_list = []
    if args.files:
        file_list = args.files
    else:
        assert args.src is not None
        for root, _, files in os.walk(args.src):
            if args.exclude and osp.realpath(root).startswith(
                    osp.realpath(args.exclude)):
                continue

            for file in files:
                if osp.splitext(file)[1] in args.suffixes:
                    file_list.append(osp.join(root, file))

    modified = False
    for file in file_list:
        suffix = osp.splitext(file)[1]
        if suffix in {'.py', '.sh'}:
            comment_symbol = '# '
        elif suffix in {'.c', '.cpp', '.cu'}:
            comment_symbol = '// '
        else:
            raise ValueError(f'Comment symbol of files with suffix {suffix} '
                             'is unspecified.')

        with open(file, 'r') as f:
            lines = f.readlines()
        if not contains_header(lines, comment_symbol, args.max_header_lines):
            if lines and lines[0].startswith('#!'):
                lines.insert(1, comment_symbol + HEADER)
            else:
                lines.insert(0, comment_symbol + HEADER)

            with open(file, 'w') as f:
                f.writelines(lines)
            modified = True

    return int(modified)


if __name__ == '__main__':
    sys.exit(main())
