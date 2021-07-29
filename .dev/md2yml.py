#!/usr/bin/env python

# This tool is used to update model-index.yml which is required by MIM, and
# will be automatically called as a pre-commit hook. The updating will be
# triggered if any change of model information (.md files in configs/) has been
# detected before a commit.

import glob
import os
import os.path as osp
import sys

import mmcv

MMSEG_ROOT = osp.dirname(osp.dirname((osp.dirname(__file__))))


def dump_yaml_and_check_difference(obj, filename):
    """Dump object to a yaml file, and check if the file content is different
    from the original.

    Args:
        obj (any): The python object to be dumped.
        filename (str): YAML filename to dump the object to.
    Returns:
        Bool: If the target YAML file is different from the original.
    """
    original = None
    if osp.isfile(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            original = f.read()
    with open(filename, 'w', encoding='utf-8') as f:
        mmcv.dump(obj, f, file_format='yaml', sort_keys=False)
    is_different = True
    if original is not None:
        with open(filename, 'r') as f:
            new = f.read()
        is_different = (original != new)
    return is_different


def parse_md(md_file):
    """Parse .md file and convert it to a .yml file which can be used for MIM.

    Args:
        md_file (str): Path to .md file.
    Returns:
        Bool: If the target YAML file is different from the original.
    """
    collection_name = osp.dirname(md_file).split('/')[-1]
    configs = os.listdir(osp.dirname(md_file))

    collection = dict(Name=collection_name, Metadata={'Training Data': []})
    models = []
    datasets = []

    with open(md_file, 'r') as md:
        lines = md.readlines()
        i = 0
        current_dataset = ''
        while i < len(lines):
            line = lines[i].strip()
            if len(line) == 0:
                i += 1
                continue
            if line[:3] == '###':
                datasets.append(line[4:])
                current_dataset = line[4:]
                i += 2
            elif line[0] == '|' and (
                    i + 1) < len(lines) and lines[i + 1][:3] == '| -':
                cols = [col.strip() for col in line.split('|')]
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
                    mem = els[mem_id] if els[mem_id] != '-' and els[
                        mem_id] != '' else -1
                    crop_size = els[crop_size_id].split('x')
                    assert len(crop_size) == 2
                    model = {
                        'Name': model_name,
                        'In Collection': collection_name,
                        'Metadata': {
                            'backbone': els[backbone_id],
                            'crop size': f'({crop_size[0]},{crop_size[1]})',
                            'lr schd': int(els[lr_schd_id]),
                        },
                        'Results': {
                            'Task': 'Semantic Segmentation',
                            'Dataset': current_dataset,
                            'Metrics': {
                                'mIoU': float(els[ss_id]),
                            },
                        },
                        'Config': config,
                        'Weights': weight,
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
                            'FP32',
                            'resolution':
                            f'({crop_size[0]},{crop_size[1]})'
                        }]
                    if mem != -1:
                        model['Metadata']['memory (GB)'] = float(mem)
                    if ms_id and els[ms_id] != '-' and els[ms_id] != '':
                        model['Results']['Metrics']['mIoU(ms+flip)'] = float(
                            els[ms_id])
                    models.append(model)
                    j += 1
                i = j
            else:
                i += 1
    collection['Metadata']['Training Data'] = datasets
    result = {'Collections': [collection], 'Models': models}
    yml_file = f'{md_file[:-9]}{collection_name}.yml'
    return dump_yaml_and_check_difference(result, yml_file)


def update_model_index():
    """Update model-index.yml according to model .md files.

    Returns:
        Bool: If the updated model-index.yml is different from the original.
    """
    configs_dir = osp.join(MMSEG_ROOT, 'configs')
    yml_files = glob.glob(osp.join(configs_dir, '**', '*.yml'), recursive=True)
    yml_files.sort()

    model_index = {
        'Import':
        [osp.relpath(yml_file, MMSEG_ROOT) for yml_file in yml_files]
    }
    model_index_file = osp.join(MMSEG_ROOT, 'model-index.yml')
    is_different = dump_yaml_and_check_difference(model_index,
                                                  model_index_file)

    return is_different


if __name__ == '__main__':
    file_list = [fn for fn in sys.argv[1:] if osp.basename(fn) == 'README.md']
    if not file_list:
        exit(0)
    file_modified = False
    for fn in file_list:
        file_modified |= parse_md(fn)

    file_modified |= update_model_index()

    exit(1 if file_modified else 0)
