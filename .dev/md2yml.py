#!/usr/bin/env python

# This tool is used to update model-index.yml which is required by MIM, and
# will be automatically called as a pre-commit hook. The updating will be
# triggered if any change of model information (.md files in configs/) has been
# detected before a commit.

# import glob
import os
import os.path as osp
# import re
import sys

MMSEG_ROOT = osp.dirname(osp.dirname((osp.dirname(__file__))))


def parse_md(md_file):
    """Parse .md file and convert it to a .yml file which can be used for MIM.

    Args:
        md_file (str): Path to .md file.
    Returns:
        Bool: If the target YAML file is different from the original.
    """
    # collection_name = osp.dirname(md_file).split('/')[-1]
    configs = os.listdir(osp.dirname(md_file)).remove(md_file)
    # collection = dict(Name=collection_name, Metadata={'Training Data': []})
    # models = []

    with open(md_file, 'read') as md:
        lines = md.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            for config in configs:
                if config in line:
                    elements = line.split('|')
                    while '' in elements:
                        elements.remove('')
                    while ',' in elements:
                        elements.remove(',')
                    last_el = elements[len(elements) - 1]
                    last_el = last_el[last_el.
                                      index('https://download.openmmlab.com'):(
                                          last_el.index('.pth') + 4)]
                    elements[len(elements) - 1] = last_el
                    elements = [el.strip() for el in elements]
                    # fps = elements[5] if elements[5] != '-' else -1
                    # model =
                    # {'Name': config, 'In Collection': collection_name}


def update_model_index():
    pass


if __name__ == '__main__':
    file_list = [fn for fn in sys.argv[1:] if osp.basename(fn) != 'README.md']
    if not file_list:
        exit(0)

    file_modified = False
    for fn in file_list:
        file_modified |= parse_md(fn)

    file_modified |= update_model_index()

    exit(1 if file_modified else 0)
