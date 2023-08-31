# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
from unittest import TestCase

import torch
from mmengine.structures import PixelData

from mmseg.evaluation import DepthMetric
from mmseg.structures import SegDataSample


class TestDepthMetric(TestCase):

    def _demo_mm_inputs(self,
                        batch_size=2,
                        image_shapes=(3, 64, 64),
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

        data_samples = []
        for idx in range(batch_size):
            image_shape = image_shapes[idx]
            _, h, w = image_shape

            data_sample = SegDataSample()
            gt_depth_map = torch.rand((1, h, w)) * 10
            data_sample.gt_depth_map = PixelData(data=gt_depth_map)

            data_samples.append(data_sample.to_dict())

        return data_samples

    def _demo_mm_model_output(self,
                              data_samples,
                              batch_size=2,
                              image_shapes=(3, 64, 64),
                              num_classes=5):

        _, h, w = image_shapes

        for data_sample in data_samples:
            data_sample['pred_depth_map'] = dict(data=torch.randn(1, h, w))

            data_sample[
                'img_path'] = 'tests/data/pseudo_dataset/imgs/00000_img.jpg'
        return data_samples

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""

        data_samples = self._demo_mm_inputs()
        data_samples = self._demo_mm_model_output(data_samples)

        depth_metric = DepthMetric()
        depth_metric.process([0] * len(data_samples), data_samples)
        res = depth_metric.compute_metrics(depth_metric.results)
        self.assertIsInstance(res, dict)

        # test save depth map file in output_dir
        depth_metric = DepthMetric(output_dir='tmp')
        depth_metric.process([0] * len(data_samples), data_samples)
        assert osp.exists('tmp')
        assert osp.isfile('tmp/00000_img.png')
        shutil.rmtree('tmp')

        # test format_only
        depth_metric = DepthMetric(output_dir='tmp', format_only=True)
        depth_metric.process([0] * len(data_samples), data_samples)
        assert depth_metric.results == []
        assert osp.exists('tmp')
        assert osp.isfile('tmp/00000_img.png')
        shutil.rmtree('tmp')
