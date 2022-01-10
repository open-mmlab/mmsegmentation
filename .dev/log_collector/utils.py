# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.dev/open-mmlab/mmcv
import os.path as osp
import sys
from importlib import import_module


def load_config(cfg_dir: str) -> dict:
    assert cfg_dir.endswith('.py')
    root_path, file_name = osp.split(cfg_dir)
    temp_module = osp.splitext(file_name)[0]
    sys.path.insert(0, root_path)
    mod = import_module(temp_module)
    sys.path.pop(0)
    cfg_dict = {
        k: v
        for k, v in mod.__dict__.items() if not k.startswith('__')
    }
    del sys.modules[temp_module]
    return cfg_dict
