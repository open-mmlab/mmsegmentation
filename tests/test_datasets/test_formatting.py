# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import numpy as np
import pytest
from mmengine.structures import BaseDataElement

from mmseg.datasets.transforms import PackSegInputs
from mmseg.structures import SegDataSample


class TestPackSegInputs(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = osp.join(osp.dirname(__file__), '../../data')
        img_path = osp.join(data_prefix, 'color.jpg')
        rng = np.random.RandomState(0)
        self.results = {
            'img_path': img_path,
            'ori_shape': (300, 400),
            'pad_shape': (600, 800),
            'img_shape': (600, 800),
            'scale_factor': 2.0,
            'flip': False,
            'flip_direction': 'horizontal',
            'img_norm_cfg': None,
            'img': rng.rand(300, 400),
            'gt_seg_map': rng.rand(300, 400),
        }
        self.meta_keys = ('img_path', 'ori_shape', 'img_shape', 'pad_shape',
                          'scale_factor', 'flip', 'flip_direction')

    def test_transform(self):
        transform = PackSegInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.results))
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], SegDataSample)
        self.assertIsInstance(results['data_samples'].gt_sem_seg,
                              BaseDataElement)
        self.assertEqual(results['data_samples'].ori_shape,
                         results['data_samples'].gt_sem_seg.shape)
        results = copy.deepcopy(self.results)
        # test dataset shape is not 2D
        results['gt_seg_map'] = np.random.rand(3, 300, 400)
        msg = 'the segmentation map is 2D'
        with pytest.warns(UserWarning, match=msg):
            results = transform(results)
        self.assertEqual(results['data_samples'].ori_shape,
                         results['data_samples'].gt_sem_seg.shape)

    def test_repr(self):
        transform = PackSegInputs(meta_keys=self.meta_keys)
        self.assertEqual(
            repr(transform), f'PackSegInputs(meta_keys={self.meta_keys})')
