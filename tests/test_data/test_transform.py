# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
from mmcv.utils import build_from_cfg
from PIL import Image

from mmseg.datasets.builder import PIPELINES


def test_resize_to_multiple():
    transform = dict(type='ResizeToMultiple', size_divisor=32)
    transform = build_from_cfg(transform, PIPELINES)

    img = np.random.randn(213, 232, 3)
    seg = np.random.randint(0, 19, (213, 232))
    results = dict()
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['pad_shape'] = img.shape

    results = transform(results)
    assert results['img'].shape == (224, 256, 3)
    assert results['gt_semantic_seg'].shape == (224, 256)
    assert results['img_shape'] == (224, 256, 3)
    assert results['pad_shape'] == (224, 256, 3)


def test_resize():
    # test assertion if img_scale is a list
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', img_scale=[1333, 800], keep_ratio=True)
        build_from_cfg(transform, PIPELINES)

    # test assertion if len(img_scale) while ratio_range is not None
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            img_scale=[(1333, 800), (1333, 600)],
            ratio_range=(0.9, 1.1),
            keep_ratio=True)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid multiscale_mode
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            img_scale=[(1333, 800), (1333, 600)],
            keep_ratio=True,
            multiscale_mode='2333')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)

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
    assert resized_results['img_shape'] == (750, 1333, 3)

    # test keep_ratio=False
    transform = dict(
        type='Resize',
        img_scale=(1280, 800),
        multiscale_mode='value',
        keep_ratio=False)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (800, 1280, 3)

    # test multiscale_mode='range'
    transform = dict(
        type='Resize',
        img_scale=[(1333, 400), (1333, 1200)],
        multiscale_mode='range',
        keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert max(resized_results['img_shape'][:2]) <= 1333
    assert min(resized_results['img_shape'][:2]) >= 400
    assert min(resized_results['img_shape'][:2]) <= 1200

    # test multiscale_mode='value'
    transform = dict(
        type='Resize',
        img_scale=[(1333, 800), (1333, 400)],
        multiscale_mode='value',
        keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] in [(750, 1333, 3), (400, 711, 3)]

    # test multiscale_mode='range'
    transform = dict(
        type='Resize',
        img_scale=(1333, 800),
        ratio_range=(0.9, 1.1),
        keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert max(resized_results['img_shape'][:2]) <= 1333 * 1.1

    # test img_scale=None and ratio_range is tuple.
    # img shape: (288, 512, 3)
    transform = dict(
        type='Resize', img_scale=None, ratio_range=(0.5, 2.0), keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert int(288 * 0.5) <= resized_results['img_shape'][0] <= 288 * 2.0
    assert int(512 * 0.5) <= resized_results['img_shape'][1] <= 512 * 2.0


def test_flip():
    # test assertion for invalid prob
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', prob=1.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', prob=1, direction='horizonta')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='RandomFlip', prob=1)
    flip_module = build_from_cfg(transform, PIPELINES)

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

    flip_module = build_from_cfg(transform, PIPELINES)
    results = flip_module(results)
    assert np.equal(original_img, results['img']).all()
    assert np.equal(original_seg, results['gt_semantic_seg']).all()


def test_random_crop():
    # test assertion for invalid random crop
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCrop', crop_size=(-1, 0))
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    seg = np.array(
        Image.open(osp.join(osp.dirname(__file__), '../data/seg.png')))
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    h, w, _ = img.shape
    transform = dict(type='RandomCrop', crop_size=(h - 20, w - 20))
    crop_module = build_from_cfg(transform, PIPELINES)
    results = crop_module(results)
    assert results['img'].shape[:2] == (h - 20, w - 20)
    assert results['img_shape'][:2] == (h - 20, w - 20)
    assert results['gt_semantic_seg'].shape[:2] == (h - 20, w - 20)


def test_pad():
    # test assertion if both size_divisor and size is None
    with pytest.raises(AssertionError):
        transform = dict(type='Pad')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='Pad', size_divisor=32)
    transform = build_from_cfg(transform, PIPELINES)
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

    resize_transform = dict(
        type='Resize', img_scale=(1333, 800), keep_ratio=True)
    resize_module = build_from_cfg(resize_transform, PIPELINES)
    results = resize_module(results)
    results = transform(results)
    img_shape = results['img'].shape
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0


def test_rotate():
    # test assertion degree should be tuple[float] or float
    with pytest.raises(AssertionError):
        transform = dict(type='RandomRotate', prob=0.5, degree=-10)
        build_from_cfg(transform, PIPELINES)
    # test assertion degree should be tuple[float] or float
    with pytest.raises(AssertionError):
        transform = dict(type='RandomRotate', prob=0.5, degree=(10., 20., 30.))
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='RandomRotate', degree=10., prob=1.)
    transform = build_from_cfg(transform, PIPELINES)

    assert str(transform) == f'RandomRotate(' \
                             f'prob={1.}, ' \
                             f'degree=({-10.}, {10.}), ' \
                             f'pad_val={0}, ' \
                             f'seg_pad_val={255}, ' \
                             f'center={None}, ' \
                             f'auto_bound={False})'

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    h, w, _ = img.shape
    seg = np.array(
        Image.open(osp.join(osp.dirname(__file__), '../data/seg.png')))
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    results = transform(results)
    assert results['img'].shape[:2] == (h, w)
    assert results['gt_semantic_seg'].shape[:2] == (h, w)


def test_normalize():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    transform = dict(type='Normalize', **img_norm_cfg)
    transform = build_from_cfg(transform, PIPELINES)
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


def test_rgb2gray():
    # test assertion out_channels should be greater than 0
    with pytest.raises(AssertionError):
        transform = dict(type='RGB2Gray', out_channels=-1)
        build_from_cfg(transform, PIPELINES)
    # test assertion weights should be tuple[float]
    with pytest.raises(AssertionError):
        transform = dict(type='RGB2Gray', out_channels=1, weights=1.1)
        build_from_cfg(transform, PIPELINES)

    # test out_channels is None
    transform = dict(type='RGB2Gray')
    transform = build_from_cfg(transform, PIPELINES)

    assert str(transform) == f'RGB2Gray(' \
                             f'out_channels={None}, ' \
                             f'weights={(0.299, 0.587, 0.114)})'

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    h, w, c = img.shape
    seg = np.array(
        Image.open(osp.join(osp.dirname(__file__), '../data/seg.png')))
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    results = transform(results)
    assert results['img'].shape == (h, w, c)
    assert results['img_shape'] == (h, w, c)
    assert results['ori_shape'] == (h, w, c)

    # test out_channels = 2
    transform = dict(type='RGB2Gray', out_channels=2)
    transform = build_from_cfg(transform, PIPELINES)

    assert str(transform) == f'RGB2Gray(' \
                             f'out_channels={2}, ' \
                             f'weights={(0.299, 0.587, 0.114)})'

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    h, w, c = img.shape
    seg = np.array(
        Image.open(osp.join(osp.dirname(__file__), '../data/seg.png')))
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    results = transform(results)
    assert results['img'].shape == (h, w, 2)
    assert results['img_shape'] == (h, w, 2)
    assert results['ori_shape'] == (h, w, c)


def test_adjust_gamma():
    # test assertion if gamma <= 0
    with pytest.raises(AssertionError):
        transform = dict(type='AdjustGamma', gamma=0)
        build_from_cfg(transform, PIPELINES)

    # test assertion if gamma is list
    with pytest.raises(AssertionError):
        transform = dict(type='AdjustGamma', gamma=[1.2])
        build_from_cfg(transform, PIPELINES)

    # test with gamma = 1.2
    transform = dict(type='AdjustGamma', gamma=1.2)
    transform = build_from_cfg(transform, PIPELINES)
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

    inv_gamma = 1.0 / 1.2
    table = np.array([((i / 255.0)**inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype('uint8')
    converted_img = mmcv.lut_transform(
        np.array(original_img, dtype=np.uint8), table)
    assert np.allclose(results['img'], converted_img)
    assert str(transform) == f'AdjustGamma(gamma={1.2})'


def test_rerange():
    # test assertion if min_value or max_value is illegal
    with pytest.raises(AssertionError):
        transform = dict(type='Rerange', min_value=[0], max_value=[255])
        build_from_cfg(transform, PIPELINES)

    # test assertion if min_value >= max_value
    with pytest.raises(AssertionError):
        transform = dict(type='Rerange', min_value=1, max_value=1)
        build_from_cfg(transform, PIPELINES)

    # test assertion if img_min_value == img_max_value
    with pytest.raises(AssertionError):
        transform = dict(type='Rerange', min_value=0, max_value=1)
        transform = build_from_cfg(transform, PIPELINES)
        results = dict()
        results['img'] = np.array([[1, 1], [1, 1]])
        transform(results)

    img_rerange_cfg = dict()
    transform = dict(type='Rerange', **img_rerange_cfg)
    transform = build_from_cfg(transform, PIPELINES)
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

    min_value = np.min(original_img)
    max_value = np.max(original_img)
    converted_img = (original_img - min_value) / (max_value - min_value) * 255

    assert np.allclose(results['img'], converted_img)
    assert str(transform) == f'Rerange(min_value={0}, max_value={255})'


def test_CLAHE():
    # test assertion if clip_limit is None
    with pytest.raises(AssertionError):
        transform = dict(type='CLAHE', clip_limit=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion if tile_grid_size is illegal
    with pytest.raises(AssertionError):
        transform = dict(type='CLAHE', tile_grid_size=(8.0, 8.0))
        build_from_cfg(transform, PIPELINES)

    # test assertion if tile_grid_size is illegal
    with pytest.raises(AssertionError):
        transform = dict(type='CLAHE', tile_grid_size=(9, 9, 9))
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='CLAHE', clip_limit=2)
    transform = build_from_cfg(transform, PIPELINES)
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

    converted_img = np.empty(original_img.shape)
    for i in range(original_img.shape[2]):
        converted_img[:, :, i] = mmcv.clahe(
            np.array(original_img[:, :, i], dtype=np.uint8), 2, (8, 8))

    assert np.allclose(results['img'], converted_img)
    assert str(transform) == f'CLAHE(clip_limit={2}, tile_grid_size={(8, 8)})'


def test_seg_rescale():
    results = dict()
    seg = np.array(
        Image.open(osp.join(osp.dirname(__file__), '../data/seg.png')))
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    h, w = seg.shape

    transform = dict(type='SegRescale', scale_factor=1. / 2)
    rescale_module = build_from_cfg(transform, PIPELINES)
    rescale_results = rescale_module(results.copy())
    assert rescale_results['gt_semantic_seg'].shape == (h // 2, w // 2)

    transform = dict(type='SegRescale', scale_factor=1)
    rescale_module = build_from_cfg(transform, PIPELINES)
    rescale_results = rescale_module(results.copy())
    assert rescale_results['gt_semantic_seg'].shape == (h, w)


def test_cutout():
    # test prob
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCutOut', prob=1.5, n_holes=1)
        build_from_cfg(transform, PIPELINES)
    # test n_holes
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut', prob=0.5, n_holes=(5, 3), cutout_shape=(8, 8))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut',
            prob=0.5,
            n_holes=(3, 4, 5),
            cutout_shape=(8, 8))
        build_from_cfg(transform, PIPELINES)
    # test cutout_shape and cutout_ratio
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut', prob=0.5, n_holes=1, cutout_shape=8)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut', prob=0.5, n_holes=1, cutout_ratio=0.2)
        build_from_cfg(transform, PIPELINES)
    # either of cutout_shape and cutout_ratio should be given
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCutOut', prob=0.5, n_holes=1)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut',
            prob=0.5,
            n_holes=1,
            cutout_shape=(2, 2),
            cutout_ratio=(0.4, 0.4))
        build_from_cfg(transform, PIPELINES)
    # test seg_fill_in
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut',
            prob=0.5,
            n_holes=1,
            cutout_shape=(8, 8),
            seg_fill_in='a')
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut',
            prob=0.5,
            n_holes=1,
            cutout_shape=(8, 8),
            seg_fill_in=256)
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')

    seg = np.array(
        Image.open(osp.join(osp.dirname(__file__), '../data/seg.png')))

    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img']

    transform = dict(
        type='RandomCutOut', prob=1, n_holes=1, cutout_shape=(10, 10))
    cutout_module = build_from_cfg(transform, PIPELINES)
    assert 'cutout_shape' in repr(cutout_module)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() < img.sum()

    transform = dict(
        type='RandomCutOut', prob=1, n_holes=1, cutout_ratio=(0.8, 0.8))
    cutout_module = build_from_cfg(transform, PIPELINES)
    assert 'cutout_ratio' in repr(cutout_module)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() < img.sum()

    transform = dict(
        type='RandomCutOut', prob=0, n_holes=1, cutout_ratio=(0.8, 0.8))
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() == img.sum()
    assert cutout_result['gt_semantic_seg'].sum() == seg.sum()

    transform = dict(
        type='RandomCutOut',
        prob=1,
        n_holes=(2, 4),
        cutout_shape=[(10, 10), (15, 15)],
        fill_in=(255, 255, 255),
        seg_fill_in=None)
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() > img.sum()
    assert cutout_result['gt_semantic_seg'].sum() == seg.sum()

    transform = dict(
        type='RandomCutOut',
        prob=1,
        n_holes=1,
        cutout_ratio=(0.8, 0.8),
        fill_in=(255, 255, 255),
        seg_fill_in=255)
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() > img.sum()
    assert cutout_result['gt_semantic_seg'].sum() > seg.sum()
