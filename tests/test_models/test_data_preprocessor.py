# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import PixelData

from mmseg.models import SegDataPreProcessor
from mmseg.structures import SegDataSample


class TestSegDataPreProcessor(TestCase):

    def test_init(self):
        # test mean is None
        processor = SegDataPreProcessor()
        self.assertTrue(not hasattr(processor, 'mean'))
        self.assertTrue(processor._enable_normalize is False)

        # test mean is not None
        processor = SegDataPreProcessor(mean=[0, 0, 0], std=[1, 1, 1])
        self.assertTrue(hasattr(processor, 'mean'))
        self.assertTrue(hasattr(processor, 'std'))
        self.assertTrue(processor._enable_normalize)

        # please specify both mean and std
        with self.assertRaises(AssertionError):
            SegDataPreProcessor(mean=[0, 0, 0])

        # bgr2rgb and rgb2bgr cannot be set to True at the same time
        with self.assertRaises(AssertionError):
            SegDataPreProcessor(bgr_to_rgb=True, rgb_to_bgr=True)

    def test_forward(self):
        data_sample = SegDataSample()
        data_sample.gt_sem_seg = PixelData(
            **{'data': torch.randint(0, 10, (1, 11, 10))})
        processor = SegDataPreProcessor(
            mean=[0, 0, 0], std=[1, 1, 1], size=(20, 20))
        data = {
            'inputs': [
                torch.randint(0, 256, (3, 11, 10)),
                torch.randint(0, 256, (3, 11, 10))
            ],
            'data_samples': [data_sample, data_sample]
        }
        out = processor(data, training=True)
        self.assertEqual(out['inputs'].shape, (2, 3, 20, 20))
        self.assertEqual(len(out['data_samples']), 2)
