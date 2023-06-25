# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
from mmengine.registry import init_default_scope
from PIL import Image

from mmseg.datasets.transforms import *  # noqa
from mmseg.datasets.transforms import (LoadBiomedicalData,
                                       LoadBiomedicalImageFromFile,
                                       PhotoMetricDistortion, RandomCrop)
from mmseg.registry import TRANSFORMS

init_default_scope('mmseg')


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
        resize_type='Resize',
        keep_ratio=False)
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
        resize_type='Resize',
        keep_ratio=False)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] in [(800, 1333), (400, 1333)]

    transform = dict(type='Resize', scale_factor=(0.9, 1.1), keep_ratio=True)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert max(resized_results['img_shape'][:2]) <= 1333 * 1.1

    # test RandomChoiceResize, which `resize_type` is `ResizeShortestEdge`
    transform = dict(
        type='RandomChoiceResize',
        scales=[128, 256, 512],
        resize_type='ResizeShortestEdge',
        max_size=1333)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'][0] in [128, 256, 512]

    transform = dict(
        type='RandomChoiceResize',
        scales=[512],
        resize_type='ResizeShortestEdge',
        max_size=512)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'][1] == 512

    transform = dict(
        type='RandomChoiceResize',
        scales=[(128, 256), (256, 512), (512, 1024)],
        resize_type='ResizeShortestEdge',
        max_size=1333)
    resize_module = TRANSFORMS.build(transform)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'][0] in [128, 256, 512]

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


def test_random_rotate_flip():
    with pytest.raises(AssertionError):
        transform = dict(type='RandomRotFlip', flip_prob=1.5)
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(type='RandomRotFlip', rotate_prob=1.5)
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(type='RandomRotFlip', degree=[20, 20, 20])
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(type='RandomRotFlip', degree=-20)
        TRANSFORMS.build(transform)

    transform = dict(
        type='RandomRotFlip', flip_prob=1.0, rotate_prob=0, degree=20)
    rot_flip_module = TRANSFORMS.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(
            osp.dirname(__file__),
            '../data/pseudo_synapse_dataset/img_dir/case0005_slice000.jpg'),
        'color')
    original_img = copy.deepcopy(img)
    seg = np.array(
        Image.open(
            osp.join(
                osp.dirname(__file__),
                '../data/pseudo_synapse_dataset/ann_dir/case0005_slice000.png')
        ))
    original_seg = copy.deepcopy(seg)
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    result_flip = rot_flip_module(results)
    assert original_img.shape == result_flip['img'].shape
    assert original_seg.shape == result_flip['gt_semantic_seg'].shape

    transform = dict(
        type='RandomRotFlip', flip_prob=0, rotate_prob=1.0, degree=20)
    rot_flip_module = TRANSFORMS.build(transform)

    result_rotate = rot_flip_module(results)
    assert original_img.shape == result_rotate['img'].shape
    assert original_seg.shape == result_rotate['gt_semantic_seg'].shape

    assert str(transform) == "{'type': 'RandomRotFlip'," \
                             " 'flip_prob': 0," \
                             " 'rotate_prob': 1.0," \
                             " 'degree': 20}"


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


def test_random_crop():
    # test assertion for invalid random crop
    with pytest.raises(AssertionError):
        RandomCrop(crop_size=(-1, 0))

    results = dict()
    img = mmcv.imread(osp.join('tests/data/color.jpg'), 'color')
    seg = np.array(Image.open(osp.join('tests/data/seg.png')))
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    h, w, _ = img.shape
    pipeline = RandomCrop(crop_size=(h - 20, w - 20))

    results = pipeline(results)
    assert results['img'].shape[:2] == (h - 20, w - 20)
    assert results['img_shape'] == (h - 20, w - 20)
    assert results['gt_semantic_seg'].shape[:2] == (h - 20, w - 20)


def test_rgb2gray():
    # test assertion out_channels should be greater than 0
    with pytest.raises(AssertionError):
        transform = dict(type='RGB2Gray', out_channels=-1)
        TRANSFORMS.build(transform)
    # test assertion weights should be tuple[float]
    with pytest.raises(AssertionError):
        transform = dict(type='RGB2Gray', out_channels=1, weights=1.1)
        TRANSFORMS.build(transform)

    # test out_channels is None
    transform = dict(type='RGB2Gray')
    transform = TRANSFORMS.build(transform)

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
    transform = TRANSFORMS.build(transform)

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


def test_photo_metric_distortion():

    results = dict()
    img = mmcv.imread(osp.join('tests/data/color.jpg'), 'color')
    seg = np.array(Image.open(osp.join('tests/data/seg.png')))
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    pipeline = PhotoMetricDistortion(saturation_range=(1., 1.))
    results = pipeline(results)

    assert (results['gt_semantic_seg'] == seg).all()
    assert results['img_shape'] == img.shape


def test_rerange():
    # test assertion if min_value or max_value is illegal
    with pytest.raises(AssertionError):
        transform = dict(type='Rerange', min_value=[0], max_value=[255])
        TRANSFORMS.build(transform)

    # test assertion if min_value >= max_value
    with pytest.raises(AssertionError):
        transform = dict(type='Rerange', min_value=1, max_value=1)
        TRANSFORMS.build(transform)

    # test assertion if img_min_value == img_max_value
    with pytest.raises(AssertionError):
        transform = dict(type='Rerange', min_value=0, max_value=1)
        transform = TRANSFORMS.build(transform)
        results = dict()
        results['img'] = np.array([[1, 1], [1, 1]])
        transform(results)

    img_rerange_cfg = dict()
    transform = dict(type='Rerange', **img_rerange_cfg)
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

    min_value = np.min(original_img)
    max_value = np.max(original_img)
    converted_img = (original_img - min_value) / (max_value - min_value) * 255

    assert np.allclose(results['img'], converted_img)
    assert str(transform) == f'Rerange(min_value={0}, max_value={255})'


def test_CLAHE():
    # test assertion if clip_limit is None
    with pytest.raises(AssertionError):
        transform = dict(type='CLAHE', clip_limit=None)
        TRANSFORMS.build(transform)

    # test assertion if tile_grid_size is illegal
    with pytest.raises(AssertionError):
        transform = dict(type='CLAHE', tile_grid_size=(8.0, 8.0))
        TRANSFORMS.build(transform)

    # test assertion if tile_grid_size is illegal
    with pytest.raises(AssertionError):
        transform = dict(type='CLAHE', tile_grid_size=(9, 9, 9))
        TRANSFORMS.build(transform)

    transform = dict(type='CLAHE', clip_limit=2)
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

    converted_img = np.empty(original_img.shape)
    for i in range(original_img.shape[2]):
        converted_img[:, :, i] = mmcv.clahe(
            np.array(original_img[:, :, i], dtype=np.uint8), 2, (8, 8))

    assert np.allclose(results['img'], converted_img)
    assert str(transform) == f'CLAHE(clip_limit={2}, tile_grid_size={(8, 8)})'


def test_adjust_gamma():
    # test assertion if gamma <= 0
    with pytest.raises(AssertionError):
        transform = dict(type='AdjustGamma', gamma=0)
        TRANSFORMS.build(transform)

    # test assertion if gamma is list
    with pytest.raises(AssertionError):
        transform = dict(type='AdjustGamma', gamma=[1.2])
        TRANSFORMS.build(transform)

    # test with gamma = 1.2
    transform = dict(type='AdjustGamma', gamma=1.2)
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

    inv_gamma = 1.0 / 1.2
    table = np.array([((i / 255.0)**inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype('uint8')
    converted_img = mmcv.lut_transform(
        np.array(original_img, dtype=np.uint8), table)
    assert np.allclose(results['img'], converted_img)
    assert str(transform) == f'AdjustGamma(gamma={1.2})'


def test_rotate():
    # test assertion degree should be tuple[float] or float
    with pytest.raises(AssertionError):
        transform = dict(type='RandomRotate', prob=0.5, degree=-10)
        TRANSFORMS.build(transform)
    # test assertion degree should be tuple[float] or float
    with pytest.raises(AssertionError):
        transform = dict(type='RandomRotate', prob=0.5, degree=(10., 20., 30.))
        TRANSFORMS.build(transform)

    transform = dict(type='RandomRotate', degree=10., prob=1.)
    transform = TRANSFORMS.build(transform)

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


def test_seg_rescale():
    results = dict()
    seg = np.array(
        Image.open(osp.join(osp.dirname(__file__), '../data/seg.png')))
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    h, w = seg.shape

    transform = dict(type='SegRescale', scale_factor=1. / 2)
    rescale_module = TRANSFORMS.build(transform)
    rescale_results = rescale_module(results.copy())
    assert rescale_results['gt_semantic_seg'].shape == (h // 2, w // 2)

    transform = dict(type='SegRescale', scale_factor=1)
    rescale_module = TRANSFORMS.build(transform)
    rescale_results = rescale_module(results.copy())
    assert rescale_results['gt_semantic_seg'].shape == (h, w)


def test_mosaic():
    # test prob
    with pytest.raises(AssertionError):
        transform = dict(type='RandomMosaic', prob=1.5)
        TRANSFORMS.build(transform)
    # test assertion for invalid img_scale
    with pytest.raises(AssertionError):
        transform = dict(type='RandomMosaic', prob=1, img_scale=640)
        TRANSFORMS.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    seg = np.array(
        Image.open(osp.join(osp.dirname(__file__), '../data/seg.png')))

    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']

    transform = dict(type='RandomMosaic', prob=1, img_scale=(10, 12))
    mosaic_module = TRANSFORMS.build(transform)
    assert 'Mosaic' in repr(mosaic_module)

    # test assertion for invalid mix_results
    with pytest.raises(AssertionError):
        mosaic_module(results)

    results['mix_results'] = [copy.deepcopy(results)] * 3
    results = mosaic_module(results)
    assert results['img'].shape[:2] == (20, 24)

    results = dict()
    results['img'] = img[:, :, 0]
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']

    transform = dict(type='RandomMosaic', prob=0, img_scale=(10, 12))
    mosaic_module = TRANSFORMS.build(transform)
    results['mix_results'] = [copy.deepcopy(results)] * 3
    results = mosaic_module(results)
    assert results['img'].shape[:2] == img.shape[:2]

    transform = dict(type='RandomMosaic', prob=1, img_scale=(10, 12))
    mosaic_module = TRANSFORMS.build(transform)
    results = mosaic_module(results)
    assert results['img'].shape[:2] == (20, 24)

    results = dict()
    results['img'] = np.concatenate((img, img), axis=2)
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']

    transform = dict(type='RandomMosaic', prob=1, img_scale=(10, 12))
    mosaic_module = TRANSFORMS.build(transform)
    results['mix_results'] = [copy.deepcopy(results)] * 3
    results = mosaic_module(results)
    assert results['img'].shape[2] == 6


def test_cutout():
    # test prob
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCutOut', prob=1.5, n_holes=1)
        TRANSFORMS.build(transform)
    # test n_holes
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut', prob=0.5, n_holes=(5, 3), cutout_shape=(8, 8))
        TRANSFORMS.build(transform)
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut',
            prob=0.5,
            n_holes=(3, 4, 5),
            cutout_shape=(8, 8))
        TRANSFORMS.build(transform)
    # test cutout_shape and cutout_ratio
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut', prob=0.5, n_holes=1, cutout_shape=8)
        TRANSFORMS.build(transform)
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut', prob=0.5, n_holes=1, cutout_ratio=0.2)
        TRANSFORMS.build(transform)
    # either of cutout_shape and cutout_ratio should be given
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCutOut', prob=0.5, n_holes=1)
        TRANSFORMS.build(transform)
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut',
            prob=0.5,
            n_holes=1,
            cutout_shape=(2, 2),
            cutout_ratio=(0.4, 0.4))
        TRANSFORMS.build(transform)
    # test seg_fill_in
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut',
            prob=0.5,
            n_holes=1,
            cutout_shape=(8, 8),
            seg_fill_in='a')
        TRANSFORMS.build(transform)
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCutOut',
            prob=0.5,
            n_holes=1,
            cutout_shape=(8, 8),
            seg_fill_in=256)
        TRANSFORMS.build(transform)

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
    cutout_module = TRANSFORMS.build(transform)
    assert 'cutout_shape' in repr(cutout_module)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() < img.sum()

    transform = dict(
        type='RandomCutOut', prob=1, n_holes=1, cutout_ratio=(0.8, 0.8))
    cutout_module = TRANSFORMS.build(transform)
    assert 'cutout_ratio' in repr(cutout_module)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() < img.sum()

    transform = dict(
        type='RandomCutOut', prob=0, n_holes=1, cutout_ratio=(0.8, 0.8))
    cutout_module = TRANSFORMS.build(transform)
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
    cutout_module = TRANSFORMS.build(transform)
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
    cutout_module = TRANSFORMS.build(transform)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() > img.sum()
    assert cutout_result['gt_semantic_seg'].sum() > seg.sum()


def test_resize_to_multiple():
    transform = dict(type='ResizeToMultiple', size_divisor=32)
    transform = TRANSFORMS.build(transform)

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
    assert results['img_shape'] == (224, 256)


def test_generate_edge():
    transform = dict(type='GenerateEdge', edge_width=1)
    transform = TRANSFORMS.build(transform)

    seg_map = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 2],
        [1, 1, 1, 2, 2],
        [1, 1, 2, 2, 2],
        [1, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ])
    results = dict()
    results['gt_seg_map'] = seg_map
    results['img_shape'] = seg_map.shape

    results = transform(results)
    assert np.all(results['gt_edge_map'] == np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]))


def test_biomedical3d_random_crop():
    # test assertion for invalid random crop
    with pytest.raises(AssertionError):
        transform = dict(type='BioMedical3DRandomCrop', crop_shape=(-2, -1, 0))
        transform = TRANSFORMS.build(transform)

    from mmseg.datasets.transforms import (LoadBiomedicalAnnotation,
                                           LoadBiomedicalImageFromFile)
    results = dict()
    results['img_path'] = osp.join(
        osp.dirname(__file__), '../data', 'biomedical.nii.gz')
    transform = LoadBiomedicalImageFromFile()
    results = transform(copy.deepcopy(results))

    results['seg_map_path'] = osp.join(
        osp.dirname(__file__), '../data', 'biomedical_ann.nii.gz')
    transform = LoadBiomedicalAnnotation()
    results = transform(copy.deepcopy(results))

    d, h, w = results['img_shape']
    transform = dict(
        type='BioMedical3DRandomCrop',
        crop_shape=(d - 20, h - 20, w - 20),
        keep_foreground=True)
    transform = TRANSFORMS.build(transform)
    crop_results = transform(results)
    assert crop_results['img'].shape[1:] == (d - 20, h - 20, w - 20)
    assert crop_results['img_shape'] == (d - 20, h - 20, w - 20)
    assert crop_results['gt_seg_map'].shape == (d - 20, h - 20, w - 20)

    transform = dict(
        type='BioMedical3DRandomCrop',
        crop_shape=(d - 20, h - 20, w - 20),
        keep_foreground=False)
    transform = TRANSFORMS.build(transform)
    crop_results = transform(results)
    assert crop_results['img'].shape[1:] == (d - 20, h - 20, w - 20)
    assert crop_results['img_shape'] == (d - 20, h - 20, w - 20)
    assert crop_results['gt_seg_map'].shape == (d - 20, h - 20, w - 20)


def test_biomedical_gaussian_noise():
    # test assertion for invalid prob
    with pytest.raises(AssertionError):
        transform = dict(type='BioMedicalGaussianNoise', prob=1.5)
        TRANSFORMS.build(transform)

    # test assertion for invalid std
    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalGaussianNoise', prob=0.2, mean=0.5, std=-0.5)
        TRANSFORMS.build(transform)

    transform = dict(type='BioMedicalGaussianNoise', prob=1.0)
    noise_module = TRANSFORMS.build(transform)
    assert str(noise_module) == 'BioMedicalGaussianNoise'\
                                '(prob=1.0, ' \
                                'mean=0.0, ' \
                                'std=0.1)'

    transform = dict(type='BioMedicalGaussianNoise', prob=1.0)
    noise_module = TRANSFORMS.build(transform)
    results = dict(
        img_path=osp.join(osp.dirname(__file__), '../data/biomedical.nii.gz'))
    from mmseg.datasets.transforms import LoadBiomedicalImageFromFile
    transform = LoadBiomedicalImageFromFile()
    results = transform(copy.deepcopy(results))
    original_img = copy.deepcopy(results['img'])
    results = noise_module(results)
    assert original_img.shape == results['img'].shape


def test_biomedical_gaussian_blur():
    # test assertion for invalid prob
    with pytest.raises(AssertionError):
        transform = dict(type='BioMedicalGaussianBlur', prob=-1.5)
        TRANSFORMS.build(transform)
    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalGaussianBlur', prob=1.0, sigma_range=0.6)
        smooth_module = TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalGaussianBlur', prob=1.0, sigma_range=(0.6))
        smooth_module = TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalGaussianBlur', prob=1.0, sigma_range=(15, 8, 9))
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalGaussianBlur', prob=1.0, sigma_range='0.16')
        TRANSFORMS.build(transform)

    transform = dict(
        type='BioMedicalGaussianBlur', prob=1.0, sigma_range=(0.7, 0.8))
    smooth_module = TRANSFORMS.build(transform)
    assert str(
        smooth_module
    ) == 'BioMedicalGaussianBlur(prob=1.0, ' \
         'prob_per_channel=0.5, '\
         'sigma_range=(0.7, 0.8), ' \
         'different_sigma_per_channel=True, '\
         'different_sigma_per_axis=True)'

    transform = dict(type='BioMedicalGaussianBlur', prob=1.0)
    smooth_module = TRANSFORMS.build(transform)
    assert str(
        smooth_module
    ) == 'BioMedicalGaussianBlur(prob=1.0, ' \
         'prob_per_channel=0.5, '\
         'sigma_range=(0.5, 1.0), ' \
         'different_sigma_per_channel=True, '\
         'different_sigma_per_axis=True)'

    results = dict(
        img_path=osp.join(osp.dirname(__file__), '../data/biomedical.nii.gz'))
    from mmseg.datasets.transforms import LoadBiomedicalImageFromFile
    transform = LoadBiomedicalImageFromFile()
    results = transform(copy.deepcopy(results))
    original_img = copy.deepcopy(results['img'])
    results = smooth_module(results)
    assert original_img.shape == results['img'].shape
    # the max value in the smoothed image should be less than the original one
    assert original_img.max() >= results['img'].max()
    assert original_img.min() <= results['img'].min()

    transform = dict(
        type='BioMedicalGaussianBlur',
        prob=1.0,
        different_sigma_per_axis=False)
    smooth_module = TRANSFORMS.build(transform)

    results = dict(
        img_path=osp.join(osp.dirname(__file__), '../data/biomedical.nii.gz'))
    from mmseg.datasets.transforms import LoadBiomedicalImageFromFile
    transform = LoadBiomedicalImageFromFile()
    results = transform(copy.deepcopy(results))
    original_img = copy.deepcopy(results['img'])
    results = smooth_module(results)
    assert original_img.shape == results['img'].shape
    # the max value in the smoothed image should be less than the original one
    assert original_img.max() >= results['img'].max()
    assert original_img.min() <= results['img'].min()


def test_BioMedicalRandomGamma():

    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalRandomGamma', prob=-1, gamma_range=(0.7, 1.2))
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalRandomGamma', prob=1.2, gamma_range=(0.7, 1.2))
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalRandomGamma', prob=1.0, gamma_range=(0.7))
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalRandomGamma',
            prob=1.0,
            gamma_range=(0.7, 0.2, 0.3))
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalRandomGamma',
            prob=1.0,
            gamma_range=(0.7, 2),
            invert_image=1)
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalRandomGamma',
            prob=1.0,
            gamma_range=(0.7, 2),
            per_channel=1)
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(
            type='BioMedicalRandomGamma',
            prob=1.0,
            gamma_range=(0.7, 2),
            retain_stats=1)
        TRANSFORMS.build(transform)

    test_img = 'tests/data/biomedical.nii.gz'
    results = dict(img_path=test_img)
    transform = LoadBiomedicalImageFromFile()
    results = transform(copy.deepcopy(results))
    origin_img = results['img']
    transform2 = dict(
        type='BioMedicalRandomGamma',
        prob=1.0,
        gamma_range=(0.7, 2),
    )
    transform2 = TRANSFORMS.build(transform2)
    results = transform2(results)
    transformed_img = results['img']
    assert origin_img.shape == transformed_img.shape


def test_BioMedical3DPad():
    # test assertion.
    with pytest.raises(AssertionError):
        transform = dict(type='BioMedical3DPad', pad_shape=None)
        TRANSFORMS.build(transform)

    with pytest.raises(AssertionError):
        transform = dict(type='BioMedical3DPad', pad_shape=[256, 256])
        TRANSFORMS.build(transform)

    data_info1 = dict(img=np.random.random((8, 6, 4, 4)))

    transform = dict(type='BioMedical3DPad', pad_shape=(6, 6, 6))
    transform = TRANSFORMS.build(transform)
    results = transform(copy.deepcopy(data_info1))
    assert results['img'].shape[1:] == (6, 6, 6)
    assert results['pad_shape'] == (6, 6, 6)

    transform = dict(type='BioMedical3DPad', pad_shape=(4, 6, 6))
    transform = TRANSFORMS.build(transform)
    results = transform(copy.deepcopy(data_info1))
    assert results['img'].shape[1:] == (6, 6, 6)
    assert results['pad_shape'] == (6, 6, 6)

    data_info2 = dict(
        img=np.random.random((8, 6, 4, 4)),
        gt_seg_map=np.random.randint(0, 2, (6, 4, 4)))

    transform = dict(type='BioMedical3DPad', pad_shape=(6, 6, 6))
    transform = TRANSFORMS.build(transform)
    results = transform(copy.deepcopy(data_info2))
    assert results['img'].shape[1:] == (6, 6, 6)
    assert results['gt_seg_map'].shape[1:] == (6, 6, 6)
    assert results['pad_shape'] == (6, 6, 6)

    transform = dict(type='BioMedical3DPad', pad_shape=(4, 6, 6))
    transform = TRANSFORMS.build(transform)
    results = transform(copy.deepcopy(data_info2))
    assert results['img'].shape[1:] == (6, 6, 6)
    assert results['gt_seg_map'].shape[1:] == (6, 6, 6)
    assert results['pad_shape'] == (6, 6, 6)


def test_biomedical_3d_flip():
    # test assertion for invalid prob
    with pytest.raises(AssertionError):
        transform = dict(type='BioMedical3DRandomFlip', prob=1.5, axes=(0, 1))
        transform = TRANSFORMS.build(transform)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(type='BioMedical3DRandomFlip', prob=1, axes=(0, 1, 3))
        transform = TRANSFORMS.build(transform)

    # test flip axes are (0, 1, 2)
    transform = dict(type='BioMedical3DRandomFlip', prob=1, axes=(0, 1, 2))
    transform = TRANSFORMS.build(transform)

    # test with random 3d data
    results = dict()
    results['img_path'] = 'Null'
    results['img_shape'] = (1, 16, 16, 16)
    results['img'] = np.random.randn(1, 16, 16, 16)
    results['gt_seg_map'] = np.random.randint(0, 4, (16, 16, 16))

    original_img = results['img'].copy()
    original_seg = results['gt_seg_map'].copy()

    # flip first time
    results = transform(results)
    with pytest.raises(AssertionError):
        assert np.equal(original_img, results['img']).all()
    with pytest.raises(AssertionError):
        assert np.equal(original_seg, results['gt_seg_map']).all()

    # flip second time
    results = transform(results)
    assert np.equal(original_img, results['img']).all()
    assert np.equal(original_seg, results['gt_seg_map']).all()

    # test with actual data and flip axes are (0, 1)
    # load biomedical 3d img and seg
    data_prefix = osp.join(osp.dirname(__file__), '../data')
    input_results = dict(img_path=osp.join(data_prefix, 'biomedical.npy'))
    biomedical_loader = LoadBiomedicalData(with_seg=True)
    data = biomedical_loader(copy.deepcopy(input_results))
    results = data.copy()

    original_img = data['img'].copy()
    original_seg = data['gt_seg_map'].copy()

    # test flip axes are (0, 1)
    transform = dict(type='BioMedical3DRandomFlip', prob=1, axes=(0, 1))
    transform = TRANSFORMS.build(transform)

    # flip first time
    results = transform(results)
    with pytest.raises(AssertionError):
        assert np.equal(original_img, results['img']).all()
    with pytest.raises(AssertionError):
        assert np.equal(original_seg, results['gt_seg_map']).all()

    # flip second time
    results = transform(results)
    assert np.equal(original_img, results['img']).all()
    assert np.equal(original_seg, results['gt_seg_map']).all()

    # test transform with flip axes = (1)
    transform = dict(type='BioMedical3DRandomFlip', prob=1, axes=(1, ))
    transform = TRANSFORMS.build(transform)
    results = data.copy()
    results = transform(results)
    results = transform(results)
    assert np.equal(original_img, results['img']).all()
    assert np.equal(original_seg, results['gt_seg_map']).all()

    # test transform with swap_label_pairs
    transform = dict(
        type='BioMedical3DRandomFlip',
        prob=1,
        axes=(1, 2),
        swap_label_pairs=[(0, 1)])
    transform = TRANSFORMS.build(transform)
    results = data.copy()
    results = transform(results)

    with pytest.raises(AssertionError):
        assert np.equal(original_seg, results['gt_seg_map']).all()

    # swap twice
    results = transform(results)
    assert np.equal(original_img, results['img']).all()
    assert np.equal(original_seg, results['gt_seg_map']).all()


def test_albu_transform():
    results = dict(
        img_path=osp.join(osp.dirname(__file__), '../data/color.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = TRANSFORMS.build(load)

    albu_transform = dict(
        type='Albu', transforms=[dict(type='ChannelShuffle', p=1)])
    albu_transform = TRANSFORMS.build(albu_transform)

    normalize = dict(type='Normalize', mean=[0] * 3, std=[0] * 3, to_rgb=True)
    normalize = TRANSFORMS.build(normalize)

    # Execute transforms
    results = load(results)
    results = albu_transform(results)
    results = normalize(results)

    assert results['img'].dtype == np.float32


def test_albu_channel_order():
    results = dict(
        img_path=osp.join(osp.dirname(__file__), '../data/color.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = TRANSFORMS.build(load)

    # Transform is modifying B channel
    albu_transform = dict(
        type='Albu',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=0,
                g_shift_limit=0,
                b_shift_limit=200,
                p=1)
        ])
    albu_transform = TRANSFORMS.build(albu_transform)

    # Execute transforms
    results_load = load(results)
    results_albu = albu_transform(results_load)

    # assert only Green and Red channel are not modified
    np.testing.assert_array_equal(results_albu['img'][..., 1:],
                                  results_load['img'][..., 1:])

    # assert Blue channel is modified
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(results_albu['img'][..., 0],
                                      results_load['img'][..., 0])
