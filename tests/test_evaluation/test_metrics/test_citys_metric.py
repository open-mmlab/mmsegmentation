# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine.structures import PixelData

from mmseg.evaluation import CityscapesMetric
from mmseg.structures import SegDataSample


class TestCityscapesMetric(TestCase):

    def _demo_mm_inputs(self,
                        batch_size=1,
                        image_shapes=(3, 128, 256),
                        num_classes=5):
        """Create a superset of inputs needed to run test or train batches.

        Args:
            batch_size (int): batch size. Default to 2.
            image_shapes (List[tuple], Optional): image shape.
                Default to (3, 64, 64)
            num_classes (int): number of different classes.
                Default to 5.
        """
        if isinstance(image_shapes, list):
            assert len(image_shapes) == batch_size
        else:
            image_shapes = [image_shapes] * batch_size

        packed_inputs = []
        for idx in range(batch_size):
            image_shape = image_shapes[idx]
            _, h, w = image_shape

            data_sample = SegDataSample()
            gt_semantic_seg = np.random.randint(
                0, num_classes, (1, h, w), dtype=np.uint8)
            gt_semantic_seg = torch.LongTensor(gt_semantic_seg)
            gt_sem_seg_data = dict(data=gt_semantic_seg)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
            data_sample = data_sample.to_dict()
            data_sample[
                'seg_map_path'] = 'tests/data/pseudo_cityscapes_dataset/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png'  # noqa
            packed_inputs.append(data_sample)

        return packed_inputs

    def _demo_mm_model_output(self,
                              batch_size=1,
                              image_shapes=(3, 128, 256),
                              num_classes=5):
        """Create a superset of inputs needed to run test or train batches.

        Args:
            batch_size (int): batch size. Default to 2.
            image_shapes (List[tuple], Optional): image shape.
                Default to (3, 64, 64)
            num_classes (int): number of different classes.
                Default to 5.
        """
        results_dict = dict()
        _, h, w = image_shapes
        seg_logit = torch.randn(batch_size, num_classes, h, w)
        results_dict['seg_logits'] = seg_logit
        seg_pred = np.random.randint(
            0, num_classes, (batch_size, h, w), dtype=np.uint8)
        seg_pred = torch.LongTensor(seg_pred)
        results_dict['pred_sem_seg'] = seg_pred

        batch_datasampes = [
            SegDataSample()
            for _ in range(results_dict['pred_sem_seg'].shape[0])
        ]
        for key, value in results_dict.items():
            for i in range(value.shape[0]):
                setattr(batch_datasampes[i], key, PixelData(data=value[i]))

        _predictions = []
        for pred in batch_datasampes:
            test_data = pred.to_dict()
            test_data[
                'img_path'] = 'tests/data/pseudo_cityscapes_dataset/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'  # noqa
            _predictions.append(test_data)

        return _predictions

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""

        data_batch = self._demo_mm_inputs(2)
        predictions = self._demo_mm_model_output(2)
        data_samples = [
            dict(**data, **result)
            for data, result in zip(data_batch, predictions)
        ]
        # test keep_results should be True when format_only is True
        with pytest.raises(AssertionError):
            CityscapesMetric(
                output_dir='tmp', format_only=True, keep_results=False)

        # test evaluate with cityscape metric
        metric = CityscapesMetric(output_dir='tmp')
        metric.process(data_batch, data_samples)
        res = metric.evaluate(2)
        self.assertIsInstance(res, dict)

        # test format_only
        metric = CityscapesMetric(
            output_dir='tmp', format_only=True, keep_results=True)
        metric.process(data_batch, data_samples)
        metric.evaluate(2)
        assert osp.exists('tmp')
        assert osp.isfile('tmp/frankfurt_000000_000294_leftImg8bit.png')
        shutil.rmtree('tmp')
