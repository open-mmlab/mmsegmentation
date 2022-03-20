# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
from os.path import dirname, exists, isdir, join, relpath

from mmcv import Config
from torch import nn

from mmseg.models import build_segmentor


def _get_config_directory():
    """Find the predefined segmentor config directory."""
    try:
        # Assume we are running in the source mmsegmentation repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmseg
        repo_dpath = dirname(dirname(mmseg.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def test_config_build_segmentor():
    """Test that all segmentation models defined in the configs can be
    initialized."""
    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    config_fpaths = []
    # one config each sub folder
    for sub_folder in os.listdir(config_dpath):
        if isdir(sub_folder):
            config_fpaths.append(
                list(glob.glob(join(config_dpath, sub_folder, '*.py')))[0])
    config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
    config_names = [relpath(p, config_dpath) for p in config_fpaths]

    print('Using {} config files'.format(len(config_names)))

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        config_mod = Config.fromfile(config_fpath)

        config_mod.model
        print('Building segmentor, config_fpath = {!r}'.format(config_fpath))

        # Remove pretrained keys to allow for testing in an offline environment
        if 'pretrained' in config_mod.model:
            config_mod.model['pretrained'] = None

        print('building {}'.format(config_fname))
        segmentor = build_segmentor(config_mod.model)
        assert segmentor is not None

        head_config = config_mod.model['decode_head']
        _check_decode_head(head_config, segmentor.decode_head)


def test_config_data_pipeline():
    """Test whether the data pipeline is valid and can process corner cases.

    CommandLine:
        xdoctest -m tests/test_config.py test_config_build_data_pipeline
    """
    import numpy as np
    from mmcv import Config

    from mmseg.datasets.pipelines import Compose

    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    import glob
    config_fpaths = list(glob.glob(join(config_dpath, '**', '*.py')))
    config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
    config_names = [relpath(p, config_dpath) for p in config_fpaths]

    print('Using {} config files'.format(len(config_names)))

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        print(
            'Building data pipeline, config_fpath = {!r}'.format(config_fpath))
        config_mod = Config.fromfile(config_fpath)

        # remove loading pipeline
        load_img_pipeline = config_mod.train_pipeline.pop(0)
        to_float32 = load_img_pipeline.get('to_float32', False)
        config_mod.train_pipeline.pop(0)
        config_mod.test_pipeline.pop(0)

        train_pipeline = Compose(config_mod.train_pipeline)
        test_pipeline = Compose(config_mod.test_pipeline)

        img = np.random.randint(0, 255, size=(1024, 2048, 3), dtype=np.uint8)
        if to_float32:
            img = img.astype(np.float32)
        seg = np.random.randint(0, 255, size=(1024, 2048, 1), dtype=np.uint8)

        results = dict(
            filename='test_img.png',
            ori_filename='test_img.png',
            img=img,
            img_shape=img.shape,
            ori_shape=img.shape,
            gt_semantic_seg=seg)
        results['seg_fields'] = ['gt_semantic_seg']

        print('Test training data pipeline: \n{!r}'.format(train_pipeline))
        output_results = train_pipeline(results)
        assert output_results is not None

        results = dict(
            filename='test_img.png',
            ori_filename='test_img.png',
            img=img,
            img_shape=img.shape,
            ori_shape=img.shape,
        )
        print('Test testing data pipeline: \n{!r}'.format(test_pipeline))
        output_results = test_pipeline(results)
        assert output_results is not None


def _check_decode_head(decode_head_cfg, decode_head):
    if isinstance(decode_head_cfg, list):
        assert isinstance(decode_head, nn.ModuleList)
        assert len(decode_head_cfg) == len(decode_head)
        num_heads = len(decode_head)
        for i in range(num_heads):
            _check_decode_head(decode_head_cfg[i], decode_head[i])
        return
    # check consistency between head_config and roi_head
    assert decode_head_cfg['type'] == decode_head.__class__.__name__

    assert decode_head_cfg['type'] == decode_head.__class__.__name__

    in_channels = decode_head_cfg.in_channels
    input_transform = decode_head.input_transform
    assert input_transform in ['resize_concat', 'multiple_select', None]
    if input_transform is not None:
        assert isinstance(in_channels, (list, tuple))
        assert isinstance(decode_head.in_index, (list, tuple))
        assert len(in_channels) == len(decode_head.in_index)
    elif input_transform == 'resize_concat':
        assert sum(in_channels) == decode_head.in_channels
    else:
        assert isinstance(in_channels, int)
        assert in_channels == decode_head.in_channels
        assert isinstance(decode_head.in_index, int)

    if decode_head_cfg['type'] == 'PointHead':
        assert decode_head_cfg.channels+decode_head_cfg.num_classes == \
               decode_head.fc_seg.in_channels
        assert decode_head.fc_seg.out_channels == decode_head_cfg.num_classes
    else:
        assert decode_head_cfg.channels == decode_head.conv_seg.in_channels
        assert decode_head.conv_seg.out_channels == decode_head_cfg.num_classes
