# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import BaseDataElement, PixelData

from mmseg.evaluation import CitysMetric
from mmseg.structures import SegDataSample


class TestCitysMetric(TestCase):

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

            mm_inputs = dict()
            data_sample = SegDataSample()
            gt_semantic_seg = np.random.randint(
                0, num_classes, (1, h, w), dtype=np.uint8)
            gt_semantic_seg = torch.LongTensor(gt_semantic_seg)
            gt_sem_seg_data = dict(data=gt_semantic_seg)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
            mm_inputs['data_sample'] = data_sample.to_dict()
            mm_inputs['data_sample']['seg_map_path'] = \
                'tests/data/pseudo_cityscapes_dataset/gtFine/val/\
                    frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png'

            mm_inputs['seg_map_path'] = mm_inputs['data_sample'][
                'seg_map_path']
            packed_inputs.append(mm_inputs)

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
            if isinstance(pred, BaseDataElement):
                test_data = pred.to_dict()
                test_data['img_path'] = \
                    'tests/data/pseudo_cityscapes_dataset/leftImg8bit/val/\
                        frankfurt/frankfurt_000000_000294_leftImg8bit.png'

                _predictions.append(test_data)
            else:
                _predictions.append(pred)
        return _predictions

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""

        data_batch = self._demo_mm_inputs(2)
        predictions = self._demo_mm_model_output(2)
        data_samples = [
            dict(**data, **result)
            for data, result in zip(data_batch, predictions)
        ]
        iou_metric = CitysMetric(citys_metrics=['cityscapes'])
        iou_metric.process(data_batch, data_samples)
        res = iou_metric.evaluate(6)
        self.assertIsInstance(res, dict)
        # test to_label_id = True
        iou_metric = CitysMetric(
            citys_metrics=['cityscapes'], to_label_id=True)
        iou_metric.process(data_batch, data_samples)
        res = iou_metric.evaluate(6)
        self.assertIsInstance(res, dict)
        import shutil
        shutil.rmtree('.format_cityscapes')
