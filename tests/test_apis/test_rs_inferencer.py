# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

import numpy as np
from mmengine import ConfigDict, init_default_scope
from utils import *  # noqa: F401, F403

from mmseg.apis import RSImage, RSInferencer
from mmseg.registry import MODELS


class TestRSImage(TestCase):

    def test_read_whole_image(self):
        init_default_scope('mmseg')
        img_path = osp.join(
            osp.dirname(__file__),
            '../data/pseudo_loveda_dataset/img_dir/0.png')
        rs_image = RSImage(img_path)
        window_size = (16, 16)
        rs_image.create_grids(window_size)
        image_data = rs_image.read(rs_image.grids[0])
        self.assertIsNotNone(image_data)

    def test_write_image_data(self):
        init_default_scope('mmseg')
        img_path = osp.join(
            osp.dirname(__file__),
            '../data/pseudo_loveda_dataset/img_dir/0.png')
        rs_image = RSImage(img_path)
        window_size = (16, 16)
        rs_image.create_grids(window_size)
        data = np.random.random((16, 16)).astype(np.int8)
        rs_image.write(data, rs_image.grids[0])


class TestRSInferencer(TestCase):

    def test_read_and_inference(self):
        init_default_scope('mmseg')
        cfg_dict = dict(
            model=dict(
                type='InferExampleModel',
                data_preprocessor=dict(type='SegDataPreProcessor'),
                backbone=dict(type='InferExampleBackbone'),
                decode_head=dict(type='InferExampleHead'),
                test_cfg=dict(mode='whole')),
            test_dataloader=dict(
                dataset=dict(
                    type='ExampleDataset',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations'),
                        dict(type='PackSegInputs')
                    ])),
            test_pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs')
            ])
        cfg = ConfigDict(cfg_dict)
        model = MODELS.build(cfg.model)
        model.cfg = cfg
        inferencer = RSInferencer.from_model(model)

        img_path = osp.join(
            osp.dirname(__file__),
            '../data/pseudo_loveda_dataset/img_dir/0.png')
        rs_image = RSImage(img_path)
        window_size = (16, 16)
        stride = (16, 16)
        inferencer.run(rs_image, window_size, stride)
