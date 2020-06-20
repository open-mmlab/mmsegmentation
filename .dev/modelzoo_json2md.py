import argparse
import os
import os.path as osp

import mmcv
from pytablewriter import Align, MarkdownTableWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Gather benchmarked models')
    parser.add_argument('table_cache', type=str, help='table_cache input')
    parser.add_argument('out', type=str, help='output path md')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    table_cache = mmcv.load(args.table_cache)
    output_dir = args.out

    writer = MarkdownTableWriter()
    writer.headers = [
        'Method', 'Backbone', 'Crop Size', 'Lr schd', 'Mem (GB)',
        'Inf time (fps)', 'mIoU', 'mIoU(ms+flip)', 'download'
    ]
    writer.margin = 1
    writer.align_list = [Align.CENTER] * len(writer.headers)
    dataset_maps = {
        'cityscapes': 'Cityscapes',
        'ade20k': 'ADE20K',
        'voc12aug': 'Pascal VOC 2012 + Aug'
    }
    for directory in table_cache:
        for dataset in table_cache[directory]:
            table = table_cache[directory][dataset][0]
            writer.table_name = dataset_maps[dataset]
            writer.value_matrix = table
            for i in range(len(table)):
                if table[i][-4] != '-':
                    table[i][-4] = f'{table[i][-4]:.2f}'
            mmcv.mkdir_or_exist(osp.join(output_dir, directory))
            writer.dump(
                osp.join(output_dir, directory, f'README_{dataset}.md'))
        with open(osp.join(output_dir, directory, 'README.md'), 'w') as dst_f:
            for dataset in dataset_maps:
                dataset_md_file = osp.join(output_dir, directory,
                                           f'README_{dataset}.md')
                with open(dataset_md_file) as src_f:
                    for line in src_f:
                        dst_f.write(line)
                    dst_f.write('\n')
                os.remove(dataset_md_file)


if __name__ == '__main__':
    main()
