# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine.structures import PixelData

from mmseg.structures import SegDataSample


def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b


class TestSegDataSample(TestCase):

    def test_init(self):
        meta_info = dict(
            img_size=[256, 256],
            scale_factor=np.array([1.5, 1.5]),
            img_shape=torch.rand(4))

        seg_data_sample = SegDataSample(metainfo=meta_info)
        assert 'img_size' in seg_data_sample
        assert seg_data_sample.img_size == [256, 256]
        assert seg_data_sample.get('img_size') == [256, 256]

    def test_setter(self):
        seg_data_sample = SegDataSample()

        # test gt_sem_seg
        gt_sem_seg_data = dict(sem_seg=torch.rand(5, 4, 2))
        gt_sem_seg = PixelData(**gt_sem_seg_data)
        seg_data_sample.gt_sem_seg = gt_sem_seg
        assert 'gt_sem_seg' in seg_data_sample
        assert _equal(seg_data_sample.gt_sem_seg.sem_seg,
                      gt_sem_seg_data['sem_seg'])

        # test pred_sem_seg
        pred_sem_seg_data = dict(sem_seg=torch.rand(5, 4, 2))
        pred_sem_seg = PixelData(**pred_sem_seg_data)
        seg_data_sample.pred_sem_seg = pred_sem_seg
        assert 'pred_sem_seg' in seg_data_sample
        assert _equal(seg_data_sample.pred_sem_seg.sem_seg,
                      pred_sem_seg_data['sem_seg'])

        # test seg_logits
        seg_logits_data = dict(sem_seg=torch.rand(5, 4, 2))
        seg_logits = PixelData(**seg_logits_data)
        seg_data_sample.seg_logits = seg_logits
        assert 'seg_logits' in seg_data_sample
        assert _equal(seg_data_sample.seg_logits.sem_seg,
                      seg_logits_data['sem_seg'])

        # test type error
        with pytest.raises(AssertionError):
            seg_data_sample.gt_sem_seg = torch.rand(2, 4)

        with pytest.raises(AssertionError):
            seg_data_sample.pred_sem_seg = torch.rand(2, 4)

        with pytest.raises(AssertionError):
            seg_data_sample.seg_logits = torch.rand(2, 4)

    def test_deleter(self):
        seg_data_sample = SegDataSample()

        pred_sem_seg_data = dict(sem_seg=torch.rand(5, 4, 2))
        pred_sem_seg = PixelData(**pred_sem_seg_data)
        seg_data_sample.pred_sem_seg = pred_sem_seg
        assert 'pred_sem_seg' in seg_data_sample
        del seg_data_sample.pred_sem_seg
        assert 'pred_sem_seg' not in seg_data_sample
