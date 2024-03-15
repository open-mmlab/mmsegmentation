#!/usr/bin/env python

# Copyright (c) OpenMMLab. All rights reserved.
# This tool is used to update model-index.yml which is required by MIM, and
# will be automatically called as a pre-commit hook. The updating will be
# triggered if any change of model information (.md files in configs/) has been
# detected before a commit.

import os
import os.path as osp
import re
import sys
from typing import List, Tuple

import yaml

MMSEG_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))


def get_collection_name_list(md_file_list: List[str]) -> List[str]:
    """Get the list of collection names."""
    collection_name_list: List[str] = []
    for md_file in md_file_list:
        with open(md_file) as f:
            lines = f.readlines()
            collection_name = lines[0].split('#')[1].strip()
            collection_name_list.append(collection_name)
    return collection_name_list


def get_md_file_list() -> Tuple[List[str], List[str]]:
    """Get the list of md files."""
    md_file_list: List[str] = []
    md_dir_list: List[str] = []
    for root, _, files in os.walk(osp.join(MMSEG_ROOT, 'configs')):
        for file in files:
            if file.endswith('.md'):
                md_file_list.append(osp.join(root, file))
                md_dir_list.append(root)
                break
    return md_file_list, md_dir_list


def get_model_info(md_file: str, config_dir: str,
                   collection_name_list: List[str]) -> Tuple[dict, str]:
    """Get model information from md file."""
    datasets: List[str] = []
    models: List[dict] = []
    current_dataset: str = ''
    paper_name: str = ''
    paper_url: str = ''
    code_url: str = ''
    is_backbone: bool = False
    is_dataset: bool = False
    collection_name: str = ''
    with open(md_file) as f:
        lines: List[str] = f.readlines()
        i: int = 0

        while i < len(lines):
            line: str = lines[i].strip()
            if len(line) == 0:
                i += 1
                continue
            # get paper name and url
            if re.match(r'> \[.*\]+\([a-zA-Z]+://[^\s]*\)', line):
                paper_info = line.split('](')
                paper_name = paper_info[0][paper_info[0].index('[') + 1:]
                paper_url = paper_info[1][:len(paper_info[1]) - 1]

            # get code info
            if 'Code Snippet' in line:
                code_url = line.split('"')[1].split('"')[0]

            if line.startswith('<!-- [BACKBONE]'):
                is_backbone = True

            if line.startswith('<!-- [DATASET]'):
                is_dataset = True

            if '<!-- [SKIP DEV CHECK] -->' in line:
                return None, None

            # get dataset names
            if line.startswith('###'):
                current_dataset = line.split('###')[1].strip()
                datasets.append(current_dataset)

            # get model info key id
            if (line[0] == '|' and (i + 1) < len(lines)
                    and lines[i + 1][:3] == '| -' and 'Method' in line
                    and 'Crop Size' in line and 'Mem (GB)' in line):
                keys: List[str] = [key.strip() for key in line.split('|')]
                crop_size_idx: int = keys.index('Crop Size')
                mem_idx: int = keys.index('Mem (GB)')
                assert 'Device' in keys, f'No Device in {md_file}'
                device_idx: int = keys.index('Device')

                if 'mIoU' in keys:
                    ss_idx = keys.index('mIoU')
                elif 'mDice' in keys:
                    ss_idx = keys.index('mDice')
                else:
                    raise ValueError(f'No mIoU or mDice in {md_file}')
                if 'mIoU(ms+flip)' in keys:
                    ms_idx = keys.index('mIoU(ms+flip)')
                elif 'Dice' in keys:
                    ms_idx = keys.index('Dice')
                else:
                    ms_idx = -1
                config_idx = keys.index('config')
                download_idx = keys.index('download')
                j: int = i + 2
                while j < len(lines) and lines[j][0] == '|':
                    values = [value.strip() for value in lines[j].split('|')]
                    # get config name
                    try:
                        config_url = re.findall(r'[a-zA-Z]+://[^\s]*py',
                                                values[config_idx])[0]
                        config_name = config_url.split('/')[-1]
                        model_name = config_name.replace('.py', '')
                    except IndexError:
                        raise ValueError(
                            f'config url is not found in {md_file}')

                    # get model name
                    try:
                        weight_url = re.findall(r'[a-zA-Z]+://[^\s]*pth',
                                                values[download_idx])[0]
                        log_url = re.findall(r'[a-zA-Z]+://[^\s]*.json',
                                             values[download_idx + 1])[0]
                    except IndexError:
                        raise ValueError(
                            f'url is not found in {values[download_idx]}')

                    # get batch size
                    bs = re.findall(r'[0-9]*xb[0-9]*',
                                    config_name)[0].split('xb')
                    batch_size = int(bs[0]) * int(bs[1])

                    # get crop size
                    crop_size = values[crop_size_idx].split('x')
                    crop_size = [int(crop_size[0]), int(crop_size[1])]

                    mem = values[mem_idx].split('\\')[0] if values[
                        mem_idx] != '-' and values[mem_idx] != '' else -1

                    method = values[keys.index('Method')].strip()
                    # method = [method.strip()] if '+' not in method else [
                    #     m.strip() for m in method.split('+')
                    # ]
                    # split method name:
                    if ' + ' in method:
                        method = [m.strip() for m in method.split(' + ')]
                    elif ' ' in method:
                        method = [m for m in method.split(' ')]
                    else:
                        method = [method]
                    backone: str = re.findall(
                        r'[^\s]*', values[keys.index('Backbone')].strip())[0]
                    archs = [backone] + method
                    collection_name = method[0]
                    config_path = osp.join('configs',
                                           config_dir.split('/')[-1],
                                           config_name)
                    model = {
                        'Name': model_name,
                        'In Collection': collection_name,
                        'Results': {
                            'Task': 'Semantic Segmentation',
                            'Dataset': current_dataset,
                            'Metrics': {
                                keys[ss_idx]: float(values[ss_idx])
                            }
                        },
                        'Config': config_path,
                        'Metadata': {
                            'Training Data':
                            current_dataset,
                            'Batch Size':
                            batch_size,
                            'Architecture':
                            archs,
                            'Training Resources':
                            f'{bs[0]}x {values[device_idx]} GPUS',
                        },
                        'Weights': weight_url,
                        'Training log': log_url,
                        'Paper': {
                            'Title': paper_name,
                            'URL': paper_url
                        },
                        'Code': code_url,
                        'Framework': 'PyTorch'
                    }
                    if ms_idx != -1 and values[ms_idx] != '-' and values[
                            ms_idx] != '':
                        model['Results']['Metrics'].update(
                            {keys[ms_idx]: float(values[ms_idx])})
                    if mem != -1:
                        model['Metadata']['Memory (GB)'] = float(mem)
                    models.append(model)
                    j += 1
                i = j
            i += 1

    if not (is_dataset
            or is_backbone) or collection_name not in collection_name_list:
        collection = {
            'Name': collection_name,
            'License': 'Apache License 2.0',
            'Metadata': {
                'Training Data': datasets
            },
            'Paper': {
                'Title': paper_name,
                'URL': paper_url,
            },
            'README': osp.join('configs',
                               config_dir.split('/')[-1], 'README.md'),
            'Frameworks': ['PyTorch'],
        }
        results = {
            'Collections': [collection],
            'Models': models
        }, collection_name
    else:
        results = {'Models': models}, ''

    return results


def dump_yaml_and_check_difference(model_info: dict, filename: str) -> bool:
    """dump yaml file and check difference with the original file.

    Args:
        model_info (dict): model info dict.
        filename (str): filename to save.
    """
    str_dump = yaml.dump(model_info, sort_keys=False)
    if osp.isfile(filename):
        file_exist = True
        with open(filename, encoding='utf-8') as f:
            str_orig = f.read()
    else:
        str_orig = None
        file_exist = False

    if file_exist and str_orig == str_dump:
        is_different = False
    else:
        is_different = True
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(str_dump)

    return is_different


def update_model_index(config_dir_list: List[str]) -> bool:
    """update model index."""
    yml_files = [
        osp.join('configs',
                 dir_name.split('/')[-1], 'metafile.yaml')
        for dir_name in config_dir_list
    ]
    yml_files.sort()

    model_index = {
        'Import': [
            osp.relpath(yml_file, MMSEG_ROOT).replace('\\', '/')
            for yml_file in yml_files
        ]
    }
    model_index_file = osp.join(MMSEG_ROOT, 'model-index.yml')
    return dump_yaml_and_check_difference(model_index, model_index_file)


if __name__ == '__main__':
    # get md file list
    md_file_list, config_dir_list = get_md_file_list()
    file_modified = False
    collection_name_list: List[str] = get_collection_name_list(md_file_list)
    # hard code to add 'FPN'
    collection_name_list.append('FPN')
    remove_config_dir_list = []
    # parse md file
    for md_file, config_dir in zip(md_file_list, config_dir_list):
        results, collection_name = get_model_info(md_file, config_dir,
                                                  collection_name_list)
        if results is None:
            remove_config_dir_list.append(config_dir)
            continue
        filename = osp.join(config_dir, 'metafile.yaml')
        file_modified |= dump_yaml_and_check_difference(results, filename)
        if collection_name != '':
            collection_name_list.append(collection_name)
    # remove config dir
    for config_dir in remove_config_dir_list:
        config_dir_list.remove(config_dir)
    file_modified |= update_model_index(config_dir_list)
    sys.exit(1 if file_modified else 0)
