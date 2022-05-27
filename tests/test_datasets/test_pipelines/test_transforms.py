# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pytest
from PIL import Image

from mmseg.datasets.pipelines import PhotoMetricDistortion, RandomCrop


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
    assert results['img_shape'][:2] == (h - 20, w - 20)
    assert results['gt_semantic_seg'].shape[:2] == (h - 20, w - 20)


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

    pipeline = PhotoMetricDistortion()
    results = pipeline(results)

    assert not ((results['img'] == img).all())
    assert (results['gt_semantic_seg'] == seg).all()
    assert results['img_shape'] == img.shape
