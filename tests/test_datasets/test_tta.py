# Copyright (c) OpenMMLab. All rights reserved.
# import os.path as osp

# import mmcv
# import pytest

# from mmseg.datasets.transforms import *  # noqa
# from mmseg.registry import TRANSFORMS

# TODO
# def test_multi_scale_flip_aug():
#     # test assertion if scales=None, scale_factor=1 (not float).
#     with pytest.raises(AssertionError):
#         tta_transform = dict(
#             type='MultiScaleFlipAug',
#             scales=None,
#             scale_factor=1,
#             transforms=[dict(type='Resize', keep_ratio=False)],
#         )
#         TRANSFORMS.build(tta_transform)

#     # test assertion if scales=None, scale_factor=None.
#     with pytest.raises(AssertionError):
#         tta_transform = dict(
#             type='MultiScaleFlipAug',
#             scales=None,
#             scale_factor=None,
#             transforms=[dict(type='Resize', keep_ratio=False)],
#         )
#         TRANSFORMS.build(tta_transform)

#     # test assertion if scales=(512, 512), scale_factor=1 (not float).
#     with pytest.raises(AssertionError):
#         tta_transform = dict(
#             type='MultiScaleFlipAug',
#             scales=(512, 512),
#             scale_factor=1,
#             transforms=[dict(type='Resize', keep_ratio=False)],
#         )
#         TRANSFORMS.build(tta_transform)
#     meta_keys = ('img', 'ori_shape', 'ori_height', 'ori_width', 'pad_shape',
#                  'scale_factor', 'scale', 'flip')
#     tta_transform = dict(
#         type='MultiScaleFlipAug',
#         scales=[(256, 256), (512, 512), (1024, 1024)],
#         allow_flip=False,
#         resize_cfg=dict(type='Resize', keep_ratio=False),
#         transforms=[dict(type='mmseg.PackSegInputs', meta_keys=meta_keys)],
#     )
#     tta_module = TRANSFORMS.build(tta_transform)

#     results = dict()
#     # (288, 512, 3)
#     img = mmcv.imread(
#         osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
#     results['img'] = img
#     results['ori_shape'] = img.shape
#     results['ori_height'] = img.shape[0]
#     results['ori_width'] = img.shape[1]
#     # Set initial values for default meta_keys
#     results['pad_shape'] = img.shape
#     results['scale_factor'] = 1.0

#     tta_results = tta_module(results.copy())
#     assert [data_sample.scale
#             for data_sample in tta_results['data_sample']] == [(256, 256),
#                                                                (512, 512),
#                                                                (1024, 1024)]
#     assert [data_sample.flip for data_sample in tta_results['data_sample']
#             ] == [False, False, False]

#     tta_transform = dict(
#         type='MultiScaleFlipAug',
#         scales=[(256, 256), (512, 512), (1024, 1024)],
#         allow_flip=True,
#         resize_cfg=dict(type='Resize', keep_ratio=False),
#         transforms=[dict(type='mmseg.PackSegInputs', meta_keys=meta_keys)],
#     )
#     tta_module = TRANSFORMS.build(tta_transform)
#     tta_results = tta_module(results.copy())
#     assert [data_sample.scale
#             for data_sample in tta_results['data_sample']] == [(256, 256),
#                                                                (256, 256),
#                                                                (512, 512),
#                                                                (512, 512),
#                                                                (1024, 1024),
#                                                                (1024, 1024)]
#     assert [data_sample.flip for data_sample in tta_results['data_sample']
#             ] == [False, True, False, True, False, True]

#     tta_transform = dict(
#         type='MultiScaleFlipAug',
#         scales=[(512, 512)],
#         allow_flip=False,
#         resize_cfg=dict(type='Resize', keep_ratio=False),
#         transforms=[dict(type='mmseg.PackSegInputs', meta_keys=meta_keys)],
#     )
#     tta_module = TRANSFORMS.build(tta_transform)
#     tta_results = tta_module(results.copy())
#     assert [tta_results['data_sample'][0].scale] == [(512, 512)]
#     assert [tta_results['data_sample'][0].flip] == [False]

#     tta_transform = dict(
#         type='MultiScaleFlipAug',
#         scales=[(512, 512)],
#         allow_flip=True,
#         resize_cfg=dict(type='Resize', keep_ratio=False),
#         transforms=[dict(type='mmseg.PackSegInputs', meta_keys=meta_keys)],
#     )
#     tta_module = TRANSFORMS.build(tta_transform)
#     tta_results = tta_module(results.copy())
#     assert [data_sample.scale
#             for data_sample in tta_results['data_sample']] == [(512, 512),
#                                                                (512, 512)]
#     assert [data_sample.flip
#             for data_sample in tta_results['data_sample']] == [False, True]

#     tta_transform = dict(
#         type='MultiScaleFlipAug',
#         scale_factor=[0.5, 1.0, 2.0],
#         allow_flip=False,
#         resize_cfg=dict(type='Resize', keep_ratio=False),
#         transforms=[dict(type='mmseg.PackSegInputs', meta_keys=meta_keys)],
#     )
#     tta_module = TRANSFORMS.build(tta_transform)
#     tta_results = tta_module(results.copy())
#     assert [data_sample.scale
#             for data_sample in tta_results['data_sample']] == [(256, 144),
#                                                                (512, 288),
#                                                                (1024, 576)]
#     assert [data_sample.flip for data_sample in tta_results['data_sample']
#             ] == [False, False, False]

#     tta_transform = dict(
#         type='MultiScaleFlipAug',
#         scale_factor=[0.5, 1.0, 2.0],
#         allow_flip=True,
#         resize_cfg=dict(type='Resize', keep_ratio=False),
#         transforms=[dict(type='mmseg.PackSegInputs', meta_keys=meta_keys)],
#     )
#     tta_module = TRANSFORMS.build(tta_transform)
#     tta_results = tta_module(results.copy())
#     assert [data_sample.scale
#             for data_sample in tta_results['data_sample']] == [(256, 144),
#                                                                (256, 144),
#                                                                (512, 288),
#                                                                (512, 288),
#                                                                (1024, 576),
#                                                                (1024, 576)]
#     assert [data_sample.flip for data_sample in tta_results['data_sample']
#             ] == [False, True, False, True, False, True]
