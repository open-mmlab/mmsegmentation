import copy
import os.path as osp

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
