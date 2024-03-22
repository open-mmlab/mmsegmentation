# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.structures import PixelData

from mmseg.engine.hooks import SegVisualizationHook
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:

        h = 288
        w = 512
        num_class = 2

        SegLocalVisualizer.get_instance('visualizer')
        SegLocalVisualizer.dataset_meta = dict(
            classes=('background', 'foreground'),
            palette=[[120, 120, 120], [6, 230, 230]])

        data_sample = SegDataSample()
        data_sample.set_metainfo({'img_path': 'tests/data/color.jpg'})
        self.data_batch = [{'data_sample': data_sample}] * 2

        pred_sem_seg_data = dict(data=torch.randint(0, num_class, (1, h, w)))
        pred_sem_seg = PixelData(**pred_sem_seg_data)
        pred_seg_data_sample = SegDataSample()
        pred_seg_data_sample.set_metainfo({'img_path': 'tests/data/color.jpg'})
        pred_seg_data_sample.pred_sem_seg = pred_sem_seg
        self.outputs = [pred_seg_data_sample] * 2

    def test_after_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = SegVisualizationHook(draw=True, interval=1)
        hook._after_iter(
            runner, 1, self.data_batch, self.outputs, mode='train')
        hook._after_iter(runner, 1, self.data_batch, self.outputs, mode='val')
        hook._after_iter(runner, 1, self.data_batch, self.outputs, mode='test')

    def test_after_val_iter(self):
        runner = Mock()
        runner.iter = 2
        hook = SegVisualizationHook(interval=1)
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

        hook = SegVisualizationHook(draw=True, interval=1)
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

        hook = SegVisualizationHook(
            draw=True, interval=1, show=True, wait_time=1)
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

    def test_after_test_iter(self):
        runner = Mock()
        hook = SegVisualizationHook(draw=True, interval=1)
        assert hook._test_index == 0
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        assert hook._test_index == len(self.outputs)
