# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from unittest import TestCase

import cv2
import mmcv
import numpy as np
import torch
from mmengine.structures import PixelData

from mmseg.structures import SegDataSample
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

        def test_add_datasample_forward(gt_sem_seg):
            data_sample = SegDataSample()
            data_sample.gt_sem_seg = gt_sem_seg

            with tempfile.TemporaryDirectory() as tmp_dir:
                seg_local_visualizer = SegLocalVisualizer(
                    vis_backends=[dict(type='LocalVisBackend')],
                    save_dir=tmp_dir)
                seg_local_visualizer.dataset_meta = dict(
                    classes=('background', 'foreground'),
                    palette=[[120, 120, 120], [6, 230, 230]])

                # test out_file
                seg_local_visualizer.add_datasample(out_file, image,
                                                    data_sample)

                assert os.path.exists(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'))
                drawn_img = cv2.imread(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'))
                assert drawn_img.shape == (h, w, 3)

                # test gt_instances and pred_instances
                pred_sem_seg_data = dict(
                    data=torch.randint(0, num_class, (1, h, w)))
                pred_sem_seg = PixelData(**pred_sem_seg_data)

                data_sample.pred_sem_seg = pred_sem_seg

                seg_local_visualizer.add_datasample(out_file, image,
                                                    data_sample)
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'), (h, w * 2, 3))

                seg_local_visualizer.add_datasample(
                    out_file, image, data_sample, draw_gt=False)
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'), (h, w, 3))

        if torch.cuda.is_available():
            test_add_datasample_forward(gt_sem_seg.cuda())
        test_add_datasample_forward(gt_sem_seg)

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

        def test_cityscapes_add_datasample_forward(gt_sem_seg):
            data_sample = SegDataSample()
            data_sample.gt_sem_seg = gt_sem_seg

            with tempfile.TemporaryDirectory() as tmp_dir:
                seg_local_visualizer = SegLocalVisualizer(
                    vis_backends=[dict(type='LocalVisBackend')],
                    save_dir=tmp_dir)
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
                # test out_file
                seg_local_visualizer.add_datasample(
                    out_file,
                    image,
                    data_sample,
                    out_file=osp.join(tmp_dir, 'test.png'))
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'test.png'), (h, w, 3))

                # test gt_instances and pred_instances
                pred_sem_seg_data = dict(
                    data=torch.randint(0, num_class, (1, h, w)))
                pred_sem_seg = PixelData(**pred_sem_seg_data)

                data_sample.pred_sem_seg = pred_sem_seg

                # test draw prediction with gt
                seg_local_visualizer.add_datasample(out_file, image,
                                                    data_sample)
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'), (h, w * 2, 3))
                # test draw prediction without gt
                seg_local_visualizer.add_datasample(
                    out_file, image, data_sample, draw_gt=False)
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'), (h, w, 3))

        if torch.cuda.is_available():
            test_cityscapes_add_datasample_forward(gt_sem_seg.cuda())
        test_cityscapes_add_datasample_forward(gt_sem_seg)

    def _assert_image_and_shape(self, out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape

    def test_add_datasample_depth(self):
        h = 10
        w = 12
        out_file = 'out_file'

        image = np.random.randint(0, 256, size=(h, w, 3)).astype('uint8')

        # test gt_depth_map
        gt_depth_map = PixelData(data=torch.rand(1, h, w))

        def test_add_datasample_forward_depth(gt_depth_map):
            data_sample = SegDataSample()
            data_sample.gt_depth_map = gt_depth_map

            with tempfile.TemporaryDirectory() as tmp_dir:
                seg_local_visualizer = SegLocalVisualizer(
                    vis_backends=[dict(type='LocalVisBackend')],
                    save_dir=tmp_dir)
                seg_local_visualizer.dataset_meta = dict(
                    classes=('background', 'foreground'),
                    palette=[[120, 120, 120], [6, 230, 230]])

                # test out_file
                seg_local_visualizer.add_datasample(out_file, image,
                                                    data_sample)

                assert os.path.exists(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'))
                drawn_img = cv2.imread(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'))
                assert drawn_img.shape == (h * 2, w, 3)

                # test gt_instances and pred_instances

                pred_depth_map = PixelData(data=torch.rand(1, h, w))

                data_sample.pred_depth_map = pred_depth_map

                seg_local_visualizer.add_datasample(out_file, image,
                                                    data_sample)
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'), (h * 2, w * 2, 3))

                seg_local_visualizer.add_datasample(
                    out_file, image, data_sample, draw_gt=False)
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'), (h * 2, w, 3))

        if torch.cuda.is_available():
            test_add_datasample_forward_depth(gt_depth_map.cuda())
        test_add_datasample_forward_depth(gt_depth_map)
