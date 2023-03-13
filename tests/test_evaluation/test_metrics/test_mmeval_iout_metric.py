# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import mmeval
import torch
from mmengine.structures import PixelData

from mmseg.evaluation import IoUMetric, MMEvalIoUMetric
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList


class TestMMEvalIoUMetric(unittest.TestCase):

    @unittest.skipIf(mmeval is None, 'MMEval is not installed.')
    def test_evalluate(self):
        dataset_metas = {
            'classes': ('person', 'rider', 'car', 'truck', 'bus', 'train',
                        'motorcycle', 'bicycle')
        }
        data_samples = self._demo_mm_inputs(1, num_classes=8)
        mmeval_iou_metric = MMEvalIoUMetric(dataset_meta=dataset_metas)
        iou_metric = IoUMetric(iou_metrics=['mIoU', 'mFscore', 'mDice'])
        iou_metric.dataset_meta = dataset_metas
        mmeval_iou_metric.process({}, data_samples)
        iou_metric.process({}, data_samples)

        results = mmeval_iou_metric.evaluate()
        iou_results = iou_metric.evaluate(len(data_samples))

        assert results['mIoU'] == iou_results['mIoU']
        assert results['aAcc'] == iou_results['aAcc']
        assert results['mAcc'] == iou_results['mAcc']
        assert results['mFscore'] == iou_results['mFscore']
        assert results['mDice'] == iou_results['mDice']
        assert results['mPrecision'] == iou_results['mPrecision']
        assert results['mRecall'] == iou_results['mRecall']

    def _demo_mm_inputs(self,
                        batch_size=2,
                        image_shapes=(3, 64, 64),
                        num_classes=5) -> SampleList:
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

        data_samples = []
        for idx in range(batch_size):
            image_shape = image_shapes[idx]
            _, h, w = image_shape
            data_sample = SegDataSample()
            preds = torch.randint(0, num_classes, (1, h, w), dtype=torch.long)
            gt_sem_seg = torch.randint(
                0, num_classes, (1, h, w), dtype=torch.long)
            data_sample.set_data({
                'pred_sem_seg':
                PixelData(**{'data': preds}),
                'gt_sem_seg':
                PixelData(**{'data': gt_sem_seg})
            })
            data_samples.append(data_sample.to_dict())

        return data_samples
