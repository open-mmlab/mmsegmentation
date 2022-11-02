# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import datetime
import json
import os
import os.path as osp
from collections import OrderedDict

from utils import load_config

# automatically collect all the results

# The structure of the directory:
#     ├── work-dir
#     │   ├── config_1
#     │   │   ├── time1.log.json
#     │   │   ├── time2.log.json
#     │   │   ├── time3.log.json
#     │   │   ├── time4.log.json
#     │   ├── config_2
#     │   │   ├── time5.log.json
#     │   │   ├── time6.log.json
#     │   │   ├── time7.log.json
#     │   │   ├── time8.log.json


def parse_args():
    parser = argparse.ArgumentParser(description='extract info from log.json')
    parser.add_argument('config_dir')
    args = parser.parse_args()
    return args


def has_keyword(name: str, keywords: list):
    for a_keyword in keywords:
        if a_keyword in name:
            return True
    return False


def main():
    args = parse_args()
    cfg = load_config(args.config_dir)
    work_dir = cfg['work_dir']
    metric = cfg['metric']
    log_items = cfg.get('log_items', [])
    ignore_keywords = cfg.get('ignore_keywords', [])
    other_info_keys = cfg.get('other_info_keys', [])
    markdown_file = cfg.get('markdown_file', None)
    json_file = cfg.get('json_file', None)

    if json_file and osp.split(json_file)[0] != '':
        os.makedirs(osp.split(json_file)[0], exist_ok=True)
    if markdown_file and osp.split(markdown_file)[0] != '':
        os.makedirs(osp.split(markdown_file)[0], exist_ok=True)

    assert not (log_items and ignore_keywords), \
        'log_items and ignore_keywords cannot be specified at the same time'
    assert metric not in other_info_keys, \
        'other_info_keys should not contain metric'

    if ignore_keywords and isinstance(ignore_keywords, str):
        ignore_keywords = [ignore_keywords]
    if other_info_keys and isinstance(other_info_keys, str):
        other_info_keys = [other_info_keys]
    if log_items and isinstance(log_items, str):
        log_items = [log_items]

    if not log_items:
        log_items = [
            item for item in sorted(os.listdir(work_dir))
            if not has_keyword(item, ignore_keywords)
        ]

    experiment_info_list = []
    for config_dir in log_items:
        preceding_path = os.path.join(work_dir, config_dir)
        log_list = [
            item for item in os.listdir(preceding_path)
            if item.endswith('.log.json')
        ]
        log_list = sorted(
            log_list,
            key=lambda time_str: datetime.datetime.strptime(
                time_str, '%Y%m%d_%H%M%S.log.json'))
        val_list = []
        last_iter = 0
        for log_name in log_list:
            with open(os.path.join(preceding_path, log_name)) as f:
                # ignore the info line
                f.readline()
                all_lines = f.readlines()
                val_list.extend([
                    json.loads(line) for line in all_lines
                    if json.loads(line)['mode'] == 'val'
                ])
                for index in range(len(all_lines) - 1, -1, -1):
                    line_dict = json.loads(all_lines[index])
                    if line_dict['mode'] == 'train':
                        last_iter = max(last_iter, line_dict['iter'])
                        break

        new_log_dict = dict(
            method=config_dir, metric_used=metric, last_iter=last_iter)
        for index, log in enumerate(val_list, 1):
            new_ordered_dict = OrderedDict()
            new_ordered_dict['eval_index'] = index
            new_ordered_dict[metric] = log[metric]
            for key in other_info_keys:
                if key in log:
                    new_ordered_dict[key] = log[key]
            val_list[index - 1] = new_ordered_dict

        assert len(val_list) >= 1, \
            f"work dir {config_dir} doesn't contain any evaluation."
        new_log_dict['last eval'] = val_list[-1]
        new_log_dict['best eval'] = max(val_list, key=lambda x: x[metric])
        experiment_info_list.append(new_log_dict)
        print(f'{config_dir} is processed')

    if json_file:
        with open(json_file, 'w') as f:
            json.dump(experiment_info_list, f, indent=4)

    if markdown_file:
        lines_to_write = []
        for index, log in enumerate(experiment_info_list, 1):
            lines_to_write.append(
                f"|{index}|{log['method']}|{log['best eval'][metric]}"
                f"|{log['best eval']['eval_index']}|"
                f"{log['last eval'][metric]}|"
                f"{log['last eval']['eval_index']}|{log['last_iter']}|\n")
        with open(markdown_file, 'w') as f:
            f.write(f'|exp_num|method|{metric} best|best index|'
                    f'{metric} last|last index|last iter num|\n')
            f.write('|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n')
            f.writelines(lines_to_write)

    print('processed successfully')


if __name__ == '__main__':
    main()
