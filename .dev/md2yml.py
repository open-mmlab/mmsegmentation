#!/usr/bin/env python

# Copyright (c) OpenMMLab. All rights reserved.
# This tool is used to update model-index.yml which is required by MIM, and
# will be automatically called as a pre-commit hook. The updating will be
# triggered if any change of model information (.md files in configs/) has been
# detected before a commit.

import glob
import os
import os.path as osp
import re
import sys

from lxml import etree
from mmcv.fileio import dump

MMSEG_ROOT = osp.dirname(osp.dirname((osp.dirname(__file__))))

COLLECTIONS = [
    'ANN', 'APCNet', 'BiSeNetV1', 'BiSeNetV2', 'CCNet', 'CGNet', 'DANet',
    'DeepLabV3', 'DeepLabV3+', 'DMNet', 'DNLNet', 'DPT', 'EMANet', 'EncNet',
    'ERFNet', 'FastFCN', 'FastSCNN', 'FCN', 'GCNet', 'ICNet', 'ISANet', 'KNet',
    'NonLocalNet', 'OCRNet', 'PointRend', 'PSANet', 'PSPNet', 'Segformer',
    'Segmenter', 'FPN', 'SETR', 'STDC', 'UNet', 'UPerNet'
]
COLLECTIONS_TEMP = []


def dump_yaml_and_check_difference(obj, filename, sort_keys=False):
    """Dump object to a yaml file, and check if the file content is different
    from the original.

    Args:
        obj (any): The python object to be dumped.
        filename (str): YAML filename to dump the object to.
        sort_keys (str); Sort key by dictionary order.
    Returns:
        Bool: If the target YAML file is different from the original.
    """

    str_dump = dump(obj, None, file_format='yaml', sort_keys=sort_keys)
    if osp.isfile(filename):
        file_exists = True
        with open(filename, 'r', encoding='utf-8') as f:
            str_orig = f.read()
    else:
        file_exists = False
        str_orig = None

    if file_exists and str_orig == str_dump:
        is_different = False
    else:
        is_different = True
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(str_dump)

    return is_different


def parse_md(md_file):
    """Parse .md file and convert it to a .yml file which can be used for MIM.

    Args:
        md_file (str): Path to .md file.
    Returns:
        Bool: If the target YAML file is different from the original.
    """
    collection_name = osp.split(osp.dirname(md_file))[1]
    configs = os.listdir(osp.dirname(md_file))

    collection = dict(
        Name=collection_name,
        Metadata={'Training Data': []},
        Paper={
            'URL': '',
            'Title': ''
        },
        README=md_file,
        Code={
            'URL': '',
            'Version': ''
        })
    collection.update({'Converted From': {'Weights': '', 'Code': ''}})
    models = []
    datasets = []
    paper_url = None
    paper_title = None
    code_url = None
    code_version = None
    repo_url = None

    # To avoid re-counting number of backbone model in OpenMMLab,
    # if certain model in configs folder is backbone whose name is already
    # recorded in MMClassification, then the `COLLECTION` dict of this model
    # in MMSegmentation should be deleted, and `In Collection` in `Models`
    # should be set with head or neck of this config file.
    is_backbone = None

    with open(md_file, 'r', encoding='UTF-8') as md:
        lines = md.readlines()
        i = 0
        current_dataset = ''
        while i < len(lines):
            line = lines[i].strip()
            # In latest README.md the title and url are in the third line.
            if i == 2:
                paper_url = lines[i].split('](')[1].split(')')[0]
                paper_title = lines[i].split('](')[0].split('[')[1]
            if len(line) == 0:
                i += 1
                continue
            elif line[:3] == '<a ':
                content = etree.HTML(line)
                node = content.xpath('//a')[0]
                if node.text == 'Code Snippet':
                    code_url = node.get('href', None)
                    assert code_url is not None, (
                        f'{collection_name} hasn\'t code snippet url.')
                    # version extraction
                    filter_str = r'blob/(.*)/mm'
                    pattern = re.compile(filter_str)
                    code_version = pattern.findall(code_url)
                    assert len(code_version) == 1, (
                        f'false regular expression ({filter_str}) use.')
                    code_version = code_version[0]
                elif node.text == 'Official Repo':
                    repo_url = node.get('href', None)
                    assert repo_url is not None, (
                        f'{collection_name} hasn\'t official repo url.')
                i += 1
            elif line[:4] == '### ':
                datasets.append(line[4:])
                current_dataset = line[4:]
                i += 2
            elif line[:15] == '<!-- [BACKBONE]':
                is_backbone = True
                i += 1
            elif (line[0] == '|' and (i + 1) < len(lines)
                  and lines[i + 1][:3] == '| -' and 'Method' in line
                  and 'Crop Size' in line and 'Mem (GB)' in line):
                cols = [col.strip() for col in line.split('|')]
                method_id = cols.index('Method')
                backbone_id = cols.index('Backbone')
                crop_size_id = cols.index('Crop Size')
                lr_schd_id = cols.index('Lr schd')
                mem_id = cols.index('Mem (GB)')
                fps_id = cols.index('Inf time (fps)')
                try:
                    ss_id = cols.index('mIoU')
                except ValueError:
                    ss_id = cols.index('Dice')
                try:
                    ms_id = cols.index('mIoU(ms+flip)')
                except ValueError:
                    ms_id = False
                config_id = cols.index('config')
                download_id = cols.index('download')
                j = i + 2
                while j < len(lines) and lines[j][0] == '|':
                    els = [el.strip() for el in lines[j].split('|')]
                    config = ''
                    model_name = ''
                    weight = ''
                    for fn in configs:
                        if fn in els[config_id]:
                            left = els[download_id].index(
                                'https://download.openmmlab.com')
                            right = els[download_id].index('.pth') + 4
                            weight = els[download_id][left:right]
                            config = f'configs/{collection_name}/{fn}'
                            model_name = fn[:-3]
                    fps = els[fps_id] if els[fps_id] != '-' and els[
                        fps_id] != '' else -1
                    mem = els[mem_id].split(
                        '\\'
                    )[0] if els[mem_id] != '-' and els[mem_id] != '' else -1
                    crop_size = els[crop_size_id].split('x')
                    assert len(crop_size) == 2
                    method = els[method_id].split()[0].split('-')[-1]
                    model = {
                        'Name':
                        model_name,
                        'In Collection':
                        method,
                        'Metadata': {
                            'backbone': els[backbone_id],
                            'crop size': f'({crop_size[0]},{crop_size[1]})',
                            'lr schd': int(els[lr_schd_id]),
                        },
                        'Results': [
                            {
                                'Task': 'Semantic Segmentation',
                                'Dataset': current_dataset,
                                'Metrics': {
                                    cols[ss_id]: float(els[ss_id]),
                                },
                            },
                        ],
                        'Config':
                        config,
                        'Weights':
                        weight,
                    }
                    if fps != -1:
                        try:
                            fps = float(fps)
                        except Exception:
                            j += 1
                            continue
                        model['Metadata']['inference time (ms/im)'] = [{
                            'value':
                            round(1000 / float(fps), 2),
                            'hardware':
                            'V100',
                            'backend':
                            'PyTorch',
                            'batch size':
                            1,
                            'mode':
                            'FP32' if 'fp16' not in config else 'FP16',
                            'resolution':
                            f'({crop_size[0]},{crop_size[1]})'
                        }]
                    if mem != -1:
                        model['Metadata']['Training Memory (GB)'] = float(mem)
                    # Only have semantic segmentation now
                    if ms_id and els[ms_id] != '-' and els[ms_id] != '':
                        model['Results'][0]['Metrics'][
                            'mIoU(ms+flip)'] = float(els[ms_id])
                    models.append(model)
                    j += 1
                i = j
            else:
                i += 1
    flag = (code_url is not None) and (paper_url is not None) and (repo_url
                                                                   is not None)
    assert flag, f'{collection_name} readme error'
    collection['Name'] = method
    collection['Metadata']['Training Data'] = datasets
    collection['Code']['URL'] = code_url
    collection['Code']['Version'] = code_version
    collection['Paper']['URL'] = paper_url
    collection['Paper']['Title'] = paper_title
    collection['Converted From']['Code'] = repo_url
    # ['Converted From']['Weights] miss
    # remove empty attribute
    check_key_list = ['Code', 'Paper', 'Converted From']
    for check_key in check_key_list:
        key_list = list(collection[check_key].keys())
        for key in key_list:
            if check_key not in collection:
                break
            if collection[check_key][key] == '':
                if len(collection[check_key].keys()) == 1:
                    collection.pop(check_key)
                else:
                    collection[check_key].pop(key)
    yml_file = f'{md_file[:-9]}{collection_name}.yml'
    if is_backbone:
        if collection['Name'] not in COLLECTIONS:
            result = {
                'Collections': [collection],
                'Models': models,
                'Yml': yml_file
            }
            COLLECTIONS_TEMP.append(result)
            return False
        else:
            result = {'Models': models}
    else:
        COLLECTIONS.append(collection['Name'])
        result = {'Collections': [collection], 'Models': models}
    return dump_yaml_and_check_difference(result, yml_file)


def update_model_index():
    """Update model-index.yml according to model .md files.

    Returns:
        Bool: If the updated model-index.yml is different from the original.
    """
    configs_dir = osp.join(MMSEG_ROOT, 'configs')
    yml_files = glob.glob(osp.join(configs_dir, '**', '*.yml'), recursive=True)
    yml_files.sort()

    # add .replace('\\', '/') to avoid Windows Style path
    model_index = {
        'Import': [
            osp.relpath(yml_file, MMSEG_ROOT).replace('\\', '/')
            for yml_file in yml_files
        ]
    }
    model_index_file = osp.join(MMSEG_ROOT, 'model-index.yml')
    is_different = dump_yaml_and_check_difference(model_index,
                                                  model_index_file)

    return is_different


if __name__ == '__main__':
    file_list = [fn for fn in sys.argv[1:] if osp.basename(fn) == 'README.md']
    if not file_list:
        sys.exit(0)
    file_modified = False
    for fn in file_list:
        file_modified |= parse_md(fn)

    for result in COLLECTIONS_TEMP:
        collection = result['Collections'][0]
        yml_file = result.pop('Yml', None)
        if collection['Name'] in COLLECTIONS:
            result.pop('Collections')
        file_modified |= dump_yaml_and_check_difference(result, yml_file)

    file_modified |= update_model_index()
    sys.exit(1 if file_modified else 0)
