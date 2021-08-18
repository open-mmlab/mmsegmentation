# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile

import mmcv
import numpy as np

from mmseg.datasets.pipelines import LoadAnnotations, LoadImageFromFile


class TestLoading(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../data')

    def test_load_img(self):
        results = dict(
            img_prefix=self.data_prefix, img_info=dict(filename='color.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['filename'] == osp.join(self.data_prefix, 'color.jpg')
        assert results['ori_filename'] == 'color.jpg'
        assert results['img'].shape == (288, 512, 3)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (288, 512, 3)
        assert results['ori_shape'] == (288, 512, 3)
        assert results['pad_shape'] == (288, 512, 3)
        assert results['scale_factor'] == 1.0
        np.testing.assert_equal(results['img_norm_cfg']['mean'],
                                np.zeros(3, dtype=np.float32))
        assert repr(transform) == transform.__class__.__name__ + \
            "(to_float32=False,color_type='color',imdecode_backend='cv2')"

        # no img_prefix
        results = dict(
            img_prefix=None, img_info=dict(filename='tests/data/color.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['filename'] == 'tests/data/color.jpg'
        assert results['ori_filename'] == 'tests/data/color.jpg'
        assert results['img'].shape == (288, 512, 3)

        # to_float32
        transform = LoadImageFromFile(to_float32=True)
        results = transform(copy.deepcopy(results))
        assert results['img'].dtype == np.float32

        # gray image
        results = dict(
            img_prefix=self.data_prefix, img_info=dict(filename='gray.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img'].shape == (288, 512, 3)
        assert results['img'].dtype == np.uint8

        transform = LoadImageFromFile(color_type='unchanged')
        results = transform(copy.deepcopy(results))
        assert results['img'].shape == (288, 512)
        assert results['img'].dtype == np.uint8
        np.testing.assert_equal(results['img_norm_cfg']['mean'],
                                np.zeros(1, dtype=np.float32))

    def test_load_seg(self):
        results = dict(
            seg_prefix=self.data_prefix,
            ann_info=dict(seg_map='seg.png'),
            seg_fields=[])
        transform = LoadAnnotations()
        results = transform(copy.deepcopy(results))
        assert results['seg_fields'] == ['gt_semantic_seg']
        assert results['gt_semantic_seg'].shape == (288, 512)
        assert results['gt_semantic_seg'].dtype == np.uint8
        assert repr(transform) == transform.__class__.__name__ + \
            "(reduce_zero_label=False,imdecode_backend='pillow')"

        # no img_prefix
        results = dict(
            seg_prefix=None,
            ann_info=dict(seg_map='tests/data/seg.png'),
            seg_fields=[])
        transform = LoadAnnotations()
        results = transform(copy.deepcopy(results))
        assert results['gt_semantic_seg'].shape == (288, 512)
        assert results['gt_semantic_seg'].dtype == np.uint8

        # reduce_zero_label
        transform = LoadAnnotations(reduce_zero_label=True)
        results = transform(copy.deepcopy(results))
        assert results['gt_semantic_seg'].shape == (288, 512)
        assert results['gt_semantic_seg'].dtype == np.uint8

        # mmcv backend
        results = dict(
            seg_prefix=self.data_prefix,
            ann_info=dict(seg_map='seg.png'),
            seg_fields=[])
        transform = LoadAnnotations(imdecode_backend='pillow')
        results = transform(copy.deepcopy(results))
        # this image is saved by PIL
        assert results['gt_semantic_seg'].shape == (288, 512)
        assert results['gt_semantic_seg'].dtype == np.uint8

    def test_load_seg_custom_classes(self):

        test_img = np.random.rand(10, 10)
        test_gt = np.zeros_like(test_img)
        test_gt[2:4, 2:4] = 1
        test_gt[2:4, 6:8] = 2
        test_gt[6:8, 2:4] = 3
        test_gt[6:8, 6:8] = 4

        tmp_dir = tempfile.TemporaryDirectory()
        img_path = osp.join(tmp_dir.name, 'img.jpg')
        gt_path = osp.join(tmp_dir.name, 'gt.png')

        mmcv.imwrite(test_img, img_path)
        mmcv.imwrite(test_gt, gt_path)

        # test only train with label with id 3
        results = dict(
            img_info=dict(filename=img_path),
            ann_info=dict(seg_map=gt_path),
            label_map={
                0: 0,
                1: 0,
                2: 0,
                3: 1,
                4: 0
            },
            seg_fields=[])

        load_imgs = LoadImageFromFile()
        results = load_imgs(copy.deepcopy(results))

        load_anns = LoadAnnotations()
        results = load_anns(copy.deepcopy(results))

        gt_array = results['gt_semantic_seg']

        true_mask = np.zeros_like(gt_array)
        true_mask[6:8, 2:4] = 1

        assert results['seg_fields'] == ['gt_semantic_seg']
        assert gt_array.shape == (10, 10)
        assert gt_array.dtype == np.uint8
        np.testing.assert_array_equal(gt_array, true_mask)

        # test only train with label with id 4 and 3
        results = dict(
            img_info=dict(filename=img_path),
            ann_info=dict(seg_map=gt_path),
            label_map={
                0: 0,
                1: 0,
                2: 0,
                3: 2,
                4: 1
            },
            seg_fields=[])

        load_imgs = LoadImageFromFile()
        results = load_imgs(copy.deepcopy(results))

        load_anns = LoadAnnotations()
        results = load_anns(copy.deepcopy(results))

        gt_array = results['gt_semantic_seg']

        true_mask = np.zeros_like(gt_array)
        true_mask[6:8, 2:4] = 2
        true_mask[6:8, 6:8] = 1

        assert results['seg_fields'] == ['gt_semantic_seg']
        assert gt_array.shape == (10, 10)
        assert gt_array.dtype == np.uint8
        np.testing.assert_array_equal(gt_array, true_mask)

        # test no custom classes
        results = dict(
            img_info=dict(filename=img_path),
            ann_info=dict(seg_map=gt_path),
            seg_fields=[])

        load_imgs = LoadImageFromFile()
        results = load_imgs(copy.deepcopy(results))

        load_anns = LoadAnnotations()
        results = load_anns(copy.deepcopy(results))

        gt_array = results['gt_semantic_seg']

        assert results['seg_fields'] == ['gt_semantic_seg']
        assert gt_array.shape == (10, 10)
        assert gt_array.dtype == np.uint8
        np.testing.assert_array_equal(gt_array, test_gt)

        tmp_dir.cleanup()
