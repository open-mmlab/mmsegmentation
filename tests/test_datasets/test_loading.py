# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.transforms import LoadImageFromFile

from mmseg.datasets.transforms import LoadAnnotations  # noqa
from mmseg.datasets.transforms import (LoadBiomedicalAnnotation,
                                       LoadBiomedicalData,
                                       LoadBiomedicalImageFromFile,
                                       LoadDepthAnnotation,
                                       LoadImageFromNDArray)


class TestLoading:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../data')

    def test_load_img(self):
        results = dict(img_path=osp.join(self.data_prefix, 'color.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img_path'] == osp.join(self.data_prefix, 'color.jpg')
        assert results['img'].shape == (288, 512, 3)
        assert results['img'].dtype == np.uint8
        assert results['ori_shape'] == results['img'].shape[:2]
        assert repr(transform) == transform.__class__.__name__ + \
               "(ignore_empty=False, to_float32=False, color_type='color'," + \
               " imdecode_backend='cv2', backend_args=None)"

        # to_float32
        transform = LoadImageFromFile(to_float32=True)
        results = transform(copy.deepcopy(results))
        assert results['img'].dtype == np.float32

        # gray image
        results = dict(img_path=osp.join(self.data_prefix, 'gray.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img'].shape == (288, 512, 3)
        assert results['img'].dtype == np.uint8

        transform = LoadImageFromFile(color_type='unchanged')
        results = transform(copy.deepcopy(results))
        assert results['img'].shape == (288, 512)
        assert results['img'].dtype == np.uint8

    def test_load_seg(self):
        seg_path = osp.join(self.data_prefix, 'seg.png')
        results = dict(
            seg_map_path=seg_path, reduce_zero_label=True, seg_fields=[])
        transform = LoadAnnotations()
        results = transform(copy.deepcopy(results))
        assert results['gt_seg_map'].shape == (288, 512)
        assert results['gt_seg_map'].dtype == np.uint8
        assert repr(transform) == transform.__class__.__name__ + \
            "(reduce_zero_label=True, imdecode_backend='pillow', " + \
            'backend_args=None)'

        # reduce_zero_label
        transform = LoadAnnotations(reduce_zero_label=True)
        results = transform(copy.deepcopy(results))
        assert results['gt_seg_map'].shape == (288, 512)
        assert results['gt_seg_map'].dtype == np.uint8

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
            img_path=img_path,
            seg_map_path=gt_path,
            label_map={
                0: 0,
                1: 0,
                2: 0,
                3: 1,
                4: 0
            },
            reduce_zero_label=False,
            seg_fields=[])

        load_imgs = LoadImageFromFile()
        results = load_imgs(copy.deepcopy(results))

        load_anns = LoadAnnotations()
        results = load_anns(copy.deepcopy(results))

        gt_array = results['gt_seg_map']

        true_mask = np.zeros_like(gt_array)
        true_mask[6:8, 2:4] = 1

        assert results['seg_fields'] == ['gt_seg_map']
        assert gt_array.shape == (10, 10)
        assert gt_array.dtype == np.uint8
        np.testing.assert_array_equal(gt_array, true_mask)

        # test only train with label with id 4 and 3
        results = dict(
            img_path=osp.join(self.data_prefix, 'color.jpg'),
            seg_map_path=gt_path,
            label_map={
                0: 0,
                1: 0,
                2: 0,
                3: 2,
                4: 1
            },
            reduce_zero_label=False,
            seg_fields=[])

        load_imgs = LoadImageFromFile()
        results = load_imgs(copy.deepcopy(results))

        load_anns = LoadAnnotations()
        results = load_anns(copy.deepcopy(results))

        gt_array = results['gt_seg_map']

        true_mask = np.zeros_like(gt_array)
        true_mask[6:8, 2:4] = 2
        true_mask[6:8, 6:8] = 1

        assert results['seg_fields'] == ['gt_seg_map']
        assert gt_array.shape == (10, 10)
        assert gt_array.dtype == np.uint8
        np.testing.assert_array_equal(gt_array, true_mask)

        # test with removing a class and reducing zero label simultaneously
        results = dict(
            img_path=img_path,
            seg_map_path=gt_path,
            # since reduce_zero_label is True, there are only 4 real classes.
            # if the full set of classes is ["A", "B", "C", "D"], the
            # following label map simulates the dataset option
            # classes=["A", "C", "D"] which removes class "B".
            label_map={
                0: 0,
                1: 255,  # simulate removing class 1
                2: 1,
                3: 2
            },
            reduce_zero_label=True,  # reduce zero label
            seg_fields=[])

        load_imgs = LoadImageFromFile()
        results = load_imgs(copy.deepcopy(results))

        # reduce zero label
        load_anns = LoadAnnotations()
        results = load_anns(copy.deepcopy(results))

        gt_array = results['gt_seg_map']

        true_mask = np.ones_like(gt_array) * 255  # all zeros get mapped to 255
        true_mask[2:4, 2:4] = 0  # 1s are reduced to class 0 mapped to class 0
        true_mask[2:4, 6:8] = 255  # 2s are reduced to class 1 which is removed
        true_mask[6:8, 2:4] = 1  # 3s are reduced to class 2 mapped to class 1
        true_mask[6:8, 6:8] = 2  # 4s are reduced to class 3 mapped to class 2

        assert results['seg_fields'] == ['gt_seg_map']
        assert gt_array.shape == (10, 10)
        assert gt_array.dtype == np.uint8
        np.testing.assert_array_equal(gt_array, true_mask)

        # test no custom classes
        results = dict(
            img_path=img_path,
            seg_map_path=gt_path,
            reduce_zero_label=False,
            seg_fields=[])

        load_imgs = LoadImageFromFile()
        results = load_imgs(copy.deepcopy(results))

        load_anns = LoadAnnotations()
        results = load_anns(copy.deepcopy(results))

        gt_array = results['gt_seg_map']

        assert results['seg_fields'] == ['gt_seg_map']
        assert gt_array.shape == (10, 10)
        assert gt_array.dtype == np.uint8
        np.testing.assert_array_equal(gt_array, test_gt)

        tmp_dir.cleanup()

    def test_load_image_from_ndarray(self):
        results = {'img': np.zeros((256, 256, 3), dtype=np.uint8)}
        transform = LoadImageFromNDArray()
        results = transform(results)

        assert results['img'].shape == (256, 256, 3)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (256, 256)
        assert results['ori_shape'] == (256, 256)

        # to_float32
        transform = LoadImageFromNDArray(to_float32=True)
        results = transform(copy.deepcopy(results))
        assert results['img'].dtype == np.float32

        # test repr
        transform = LoadImageFromNDArray()
        assert repr(transform) == ('LoadImageFromNDArray('
                                   'ignore_empty=False, '
                                   'to_float32=False, '
                                   "color_type='color', "
                                   "imdecode_backend='cv2', "
                                   'backend_args=None)')

    def test_load_biomedical_img(self):
        results = dict(
            img_path=osp.join(self.data_prefix, 'biomedical.nii.gz'))
        transform = LoadBiomedicalImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img_path'] == osp.join(self.data_prefix,
                                               'biomedical.nii.gz')
        assert len(results['img'].shape) == 4
        assert results['img'].dtype == np.float32
        assert results['ori_shape'] == results['img'].shape[1:]
        assert repr(transform) == ('LoadBiomedicalImageFromFile('
                                   "decode_backend='nifti', "
                                   'to_xyz=False, '
                                   'to_float32=True, '
                                   'backend_args=None)')

    def test_load_biomedical_annotation(self):
        results = dict(
            seg_map_path=osp.join(self.data_prefix, 'biomedical_ann.nii.gz'))
        transform = LoadBiomedicalAnnotation()
        results = transform(copy.deepcopy(results))
        assert len(results['gt_seg_map'].shape) == 3
        assert results['gt_seg_map'].dtype == np.float32

    def test_load_biomedical_data(self):
        input_results = dict(
            img_path=osp.join(self.data_prefix, 'biomedical.npy'))
        transform = LoadBiomedicalData(with_seg=True)
        results = transform(copy.deepcopy(input_results))
        assert results['img_path'] == osp.join(self.data_prefix,
                                               'biomedical.npy')
        assert results['img'][0].shape == results['gt_seg_map'].shape
        assert results['img'].dtype == np.float32
        assert results['ori_shape'] == results['img'].shape[1:]
        assert repr(transform) == ('LoadBiomedicalData('
                                   'with_seg=True, '
                                   "decode_backend='numpy', "
                                   'to_xyz=False, '
                                   'backend_args=None)')

        transform = LoadBiomedicalData(with_seg=False)
        results = transform(copy.deepcopy(input_results))
        assert len(results['img'].shape) == 4
        assert results.get('gt_seg_map') is None
        assert repr(transform) == ('LoadBiomedicalData('
                                   'with_seg=False, '
                                   "decode_backend='numpy', "
                                   'to_xyz=False, '
                                   'backend_args=None)')

    def test_load_depth_annotation(self):
        input_results = dict(
            img_path='tests/data/pseudo_nyu_dataset/images/'
            'bookstore_0001d_00001.jpg',
            depth_map_path='tests/data/pseudo_nyu_dataset/'
            'annotations/bookstore_0001d_00001.png',
            category_id=-1,
            seg_fields=[])
        transform = LoadDepthAnnotation(depth_rescale_factor=0.001)
        results = transform(input_results)
        assert 'gt_depth_map' in results
        assert results['gt_depth_map'].shape[:2] == mmcv.imread(
            input_results['depth_map_path']).shape[:2]
        assert results['gt_depth_map'].dtype == np.float32
        assert 'gt_depth_map' in results['seg_fields']
