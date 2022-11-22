# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import pytest

from mmseg.datasets.transforms import *  # noqa
from mmseg.registry import TRANSFORMS


def test_multi_scale_flip_aug():
    # test exception
    with pytest.raises(TypeError):
        tta_transform = dict(
            type='TestTimeAug',
            transforms=[dict(type='Resize', keep_ratio=False)],
        )
        TRANSFORMS.build(tta_transform)

    tta_transform = dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=scale, keep_ratio=False)
            for scale in [(256, 256), (512, 512), (1024, 1024)]
        ], [dict(type='mmseg.PackSegInputs')]])
    tta_module = TRANSFORMS.build(tta_transform)

    results = dict()
    # (288, 512, 3)
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    results['img'] = img
    results['ori_shape'] = img.shape
    results['ori_height'] = img.shape[0]
    results['ori_width'] = img.shape[1]
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    tta_results = tta_module(results.copy())
    assert [img.shape for img in tta_results['inputs']] == [(3, 256, 256),
                                                            (3, 512, 512),
                                                            (3, 1024, 1024)]

    tta_transform = dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=scale, keep_ratio=False)
                for scale in [(256, 256), (512, 512), (1024, 1024)]
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='mmseg.PackSegInputs')]
        ])
    tta_module = TRANSFORMS.build(tta_transform)
    tta_results: dict = tta_module(results.copy())
    assert [img.shape for img in tta_results['inputs']] == [(3, 256, 256),
                                                            (3, 256, 256),
                                                            (3, 512, 512),
                                                            (3, 512, 512),
                                                            (3, 1024, 1024),
                                                            (3, 1024, 1024)]
    assert [
        data_sample.metainfo['flip']
        for data_sample in tta_results['data_samples']
    ] == [False, True, False, True, False, True]

    tta_transform = dict(
        type='TestTimeAug',
        transforms=[[dict(type='Resize', scale=(512, 512), keep_ratio=False)],
                    [dict(type='mmseg.PackSegInputs')]])
    tta_module = TRANSFORMS.build(tta_transform)
    tta_results = tta_module(results.copy())
    assert [tta_results['inputs'][0].shape] == [(3, 512, 512)]

    tta_transform = dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale=(512, 512), keep_ratio=False)],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='mmseg.PackSegInputs')]
        ])
    tta_module = TRANSFORMS.build(tta_transform)
    tta_results = tta_module(results.copy())
    assert [img.shape for img in tta_results['inputs']] == [(3, 512, 512),
                                                            (3, 512, 512)]
    assert [
        data_sample.metainfo['flip']
        for data_sample in tta_results['data_samples']
    ] == [False, True]

    tta_transform = dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale_factor=r, keep_ratio=False)
            for r in [0.5, 1.0, 2.0]
        ], [dict(type='mmseg.PackSegInputs')]])
    tta_module = TRANSFORMS.build(tta_transform)
    tta_results = tta_module(results.copy())
    assert [img.shape for img in tta_results['inputs']] == [(3, 144, 256),
                                                            (3, 288, 512),
                                                            (3, 576, 1024)]

    tta_transform = dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in [0.5, 1.0, 2.0]
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='mmseg.PackSegInputs')]
        ])
    tta_module = TRANSFORMS.build(tta_transform)
    tta_results = tta_module(results.copy())
    assert [img.shape for img in tta_results['inputs']] == [(3, 144, 256),
                                                            (3, 144, 256),
                                                            (3, 288, 512),
                                                            (3, 288, 512),
                                                            (3, 576, 1024),
                                                            (3, 576, 1024)]
    assert [
        data_sample.metainfo['flip']
        for data_sample in tta_results['data_samples']
    ] == [False, True, False, True, False, True]
