# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from unittest import TestCase

import cv2
import mmcv
import numpy as np
import pytest
import torch
from mmengine.data import PixelData

from mmseg.data import SegDataSample
from mmseg.visualization import SegLocalVisualizer


class TestSegLocalVisualizer(TestCase):

    def test_add_datasample(self):
        h = 10
        w = 12
        num_class = 2
        out_file = 'out_file'

        image = np.random.randint(0, 256, size=(h, w, 3)).astype('uint8')

        # test gt_sem_seg
        gt_sem_seg_data = dict(data=torch.randint(0, num_class, (1, h, w)))
        gt_sem_seg = PixelData(**gt_sem_seg_data)

        @pytest.mark.parametrize('gt_sem_seg', (gt_sem_seg, gt_sem_seg.cuda()))
        def test_add_datasample_forward(gt_sem_seg):
            gt_seg_data_sample = SegDataSample()
            gt_seg_data_sample.gt_sem_seg = gt_sem_seg

            with tempfile.TemporaryDirectory(dir='temp_dir') as tmp_dir:
                seg_local_visualizer = SegLocalVisualizer(
                    vis_backends=[dict(type='LocalVisBackend')],
                    save_dir=tmp_dir)
                seg_local_visualizer.dataset_meta = dict(
                    classes=('background', 'foreground'),
                    palette=[[120, 120, 120], [6, 230, 230]])

                # test out_file
                seg_local_visualizer.add_datasample(out_file, image,
                                                    gt_seg_data_sample)

                assert os.path.exists(
                    osp.join(tmp_dir + '/vis_data/vis_image',
                             out_file + '_0.png'))
                drawn_img = cv2.imread(
                    osp.join(tmp_dir + '/vis_data/vis_image',
                             out_file + '_0.png'))
                assert drawn_img.shape == (h, w, 3)

                # test gt_instances and pred_instances
                pred_sem_seg_data = dict(
                    data=torch.randint(0, num_class, (1, h, w)))
                pred_sem_seg = PixelData(**pred_sem_seg_data)

                pred_seg_data_sample = SegDataSample()
                pred_seg_data_sample.pred_sem_seg = pred_sem_seg

                seg_local_visualizer.add_datasample(out_file, image,
                                                    gt_seg_data_sample,
                                                    pred_seg_data_sample)
                self._assert_image_and_shape(
                    osp.join(tmp_dir + '/vis_data/vis_image',
                             out_file + '_0.png'), (h, w * 2, 3))

                seg_local_visualizer.add_datasample(
                    out_file,
                    image,
                    gt_seg_data_sample,
                    pred_seg_data_sample,
                    draw_gt=False)
                self._assert_image_and_shape(
                    osp.join(tmp_dir + '/vis_data/vis_image',
                             out_file + '_0.png'), (h, w, 3))

                seg_local_visualizer.add_datasample(
                    out_file,
                    image,
                    gt_seg_data_sample,
                    pred_seg_data_sample,
                    draw_pred=False)
                self._assert_image_and_shape(
                    osp.join(tmp_dir + '/vis_data/vis_image',
                             out_file + '_0.png'), (h, w, 3))

    def test_cityscapes_add_datasample(self):
        h = 128
        w = 256
        num_class = 19
        out_file = 'out_file_cityscapes'

        image = mmcv.imread(
            osp.join(
                osp.dirname(__file__),
                '../data/pseudo_cityscapes_dataset/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'  # noqa
            ),
            'color')
        sem_seg = mmcv.imread(
            osp.join(
                osp.dirname(__file__),
                '../data/pseudo_cityscapes_dataset/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png'  # noqa
            ),
            'unchanged')
        sem_seg = torch.unsqueeze(torch.from_numpy(sem_seg), 0)
        gt_sem_seg_data = dict(data=sem_seg)
        gt_sem_seg = PixelData(**gt_sem_seg_data)

        @pytest.mark.parametrize('gt_sem_seg', (gt_sem_seg, gt_sem_seg.cuda()))
        def test_cityscapes_add_datasample_forward(gt_sem_seg):
            gt_seg_data_sample = SegDataSample()
            gt_seg_data_sample.gt_sem_seg = gt_sem_seg
            with tempfile.TemporaryDirectory(dir='temp_dir') as tmp_dir:
                seg_local_visualizer = SegLocalVisualizer(
                    vis_backends=[dict(type='LocalVisBackend')],
                    save_dir='temp_dir')
                seg_local_visualizer.dataset_meta = dict(
                    classes=('road', 'sidewalk', 'building', 'wall', 'fence',
                             'pole', 'traffic light', 'traffic sign',
                             'vegetation', 'terrain', 'sky', 'person', 'rider',
                             'car', 'truck', 'bus', 'train', 'motorcycle',
                             'bicycle'),
                    palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70],
                             [102, 102, 156], [190, 153, 153], [153, 153, 153],
                             [250, 170, 30], [220, 220, 0], [107, 142, 35],
                             [152, 251, 152], [70, 130, 180], [220, 20, 60],
                             [255, 0, 0], [0, 0, 142], [0, 0, 70],
                             [0, 60, 100], [0, 80, 100], [0, 0, 230],
                             [119, 11, 32]])
                seg_local_visualizer.add_datasample(out_file, image,
                                                    gt_seg_data_sample)

                # test out_file
                seg_local_visualizer.add_datasample(out_file, image,
                                                    gt_seg_data_sample)
                assert os.path.exists(
                    osp.join(tmp_dir + '/vis_data/vis_image',
                             out_file + '_0.png'))
                drawn_img = cv2.imread(
                    osp.join(tmp_dir + '/vis_data/vis_image',
                             out_file + '_0.png'))
                assert drawn_img.shape == (h, w, 3)

                # test gt_instances and pred_instances
                pred_sem_seg_data = dict(
                    data=torch.randint(0, num_class, (1, h, w)))
                pred_sem_seg = PixelData(**pred_sem_seg_data)

                pred_seg_data_sample = SegDataSample()
                pred_seg_data_sample.pred_sem_seg = pred_sem_seg

                seg_local_visualizer.add_datasample(out_file, image,
                                                    gt_seg_data_sample,
                                                    pred_seg_data_sample)
                self._assert_image_and_shape(
                    osp.join(tmp_dir + '/vis_data/vis_image',
                             out_file + '_0.png'), (h, w * 2, 3))

                seg_local_visualizer.add_datasample(
                    out_file,
                    image,
                    gt_seg_data_sample,
                    pred_seg_data_sample,
                    draw_gt=False)
                self._assert_image_and_shape(
                    osp.join(tmp_dir + '/vis_data/vis_image',
                             out_file + '_0.png'), (h, w, 3))

                seg_local_visualizer.add_datasample(
                    out_file,
                    image,
                    gt_seg_data_sample,
                    pred_seg_data_sample,
                    draw_pred=False)
                self._assert_image_and_shape(
                    osp.join(tmp_dir + '/vis_data/vis_image',
                             out_file + '_0.png'), (h, w, 3))

    def _assert_image_and_shape(self, out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape
