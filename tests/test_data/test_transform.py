# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
from PIL import Image

from mmseg.registry import TRANSFORMS


def test_resize():
    # Test `Resize`, `RandomResize` and `RandomChoiceResize` from
    # MMCV transform. Noted: `RandomResize` has args `scales` but
    # `Resize` and `RandomResize` has args `scale`.
    transform = dict(type='Resize', scale=(1333, 800), keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)

    results = dict()
    # (288, 512, 3)
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    resized_results = resize_module(results.copy())
    # img_shape = results['img'].shape[:2] in ``MMCV resize`` function
    # so right now it is (750, 1333) rather than (750, 1333, 3)
    assert resized_results['img_shape'] == (750, 1333)

    # test keep_ratio=False
    transform = dict(
        type='RandomResize',
        scale=(1280, 800),
        ratio_range=(1.0, 1.0),
        resize_cfg=dict(type='Resize', keep_ratio=False))
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (800, 1280)

    # test `RandomChoiceResize`, which in older mmsegmentation
    # `Resize` is multiscale_mode='range'
    transform = dict(type='RandomResize', scale=[(1333, 400), (1333, 1200)])
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert max(resized_results['img_shape'][:2]) <= 1333
    assert min(resized_results['img_shape'][:2]) >= 400
    assert min(resized_results['img_shape'][:2]) <= 1200

    # test RandomChoiceResize, which in older mmsegmentation
    # `Resize` is multiscale_mode='value'
    transform = dict(
        type='RandomChoiceResize',
        scales=[(1333, 800), (1333, 400)],
        resize_cfg=dict(type='Resize', keep_ratio=True))
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] in [(750, 1333), (400, 711)]

    transform = dict(type='Resize', scale_factor=(0.9, 1.1), keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert max(resized_results['img_shape'][:2]) <= 1333 * 1.1

    # test scale=None and scale_factor is tuple.
    # img shape: (288, 512, 3)
    transform = dict(
        type='Resize', scale=None, scale_factor=(0.5, 2.0), keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert int(288 * 0.5) <= resized_results['img_shape'][0] <= 288 * 2.0
    assert int(512 * 0.5) <= resized_results['img_shape'][1] <= 512 * 2.0

    # test minimum resized image shape is 640
    transform = dict(type='Resize', scale=(2560, 640), keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (640, 1138)

    # test minimum resized image shape is 640 when img_scale=(512, 640)
    # where should define `scale_factor` in MMCV new ``Resize`` function.
    min_size_ratio = max(640 / img.shape[0], 640 / img.shape[1])
    transform = dict(
        type='Resize', scale_factor=min_size_ratio, keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (640, 1138)

    # test h > w
    img = np.random.randn(512, 288, 3)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    min_size_ratio = max(640 / img.shape[0], 640 / img.shape[1])
    transform = dict(
        type='Resize',
        scale=(2560, 640),
        scale_factor=min_size_ratio,
        keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (1138, 640)


def test_flip():
    # test assertion for invalid prob
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', prob=1.5)
        TRANSFORMS.build(transform)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', prob=1.0, direction='horizonta')
        TRANSFORMS.build(transform)

    transform = dict(type='RandomFlip', prob=1.0)
    flip_module = TRANSFORMS.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    seg = np.array(
        Image.open(osp.join(osp.dirname(__file__), '../data/seg.png')))
    original_seg = copy.deepcopy(seg)
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    results = flip_module(results)

    flip_module = TRANSFORMS.build(transform)
    results = flip_module(results)
    assert np.equal(original_img, results['img']).all()
    assert np.equal(original_seg, results['gt_semantic_seg']).all()


def test_pad():
    # test assertion if both size_divisor and size is None
    with pytest.raises(AssertionError):
        transform = dict(type='Pad')
        TRANSFORMS.build(transform)

    transform = dict(type='Pad', size_divisor=32)
    transform = TRANSFORMS.build(transform)
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    results = transform(results)
    # original img already divisible by 32
    assert np.equal(results['img'], original_img).all()
    img_shape = results['img'].shape
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0


def test_normalize():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    transform = dict(type='Normalize', **img_norm_cfg)
    transform = TRANSFORMS.build(transform)
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    results = transform(results)

    mean = np.array(img_norm_cfg['mean'])
    std = np.array(img_norm_cfg['std'])
    converted_img = (original_img[..., ::-1] - mean) / std
    assert np.allclose(results['img'], converted_img)
