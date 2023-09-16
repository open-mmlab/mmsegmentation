# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from os.path import dirname, join
from unittest import TestCase

import torch
from mmengine import Config, ConfigDict
from mmengine.structures import PixelData

import mmseg
from mmseg.models.segmentors import DepthEstimator
from mmseg.structures import SegDataSample


class TestDepthEstimator(TestCase):

    def setUp(self) -> None:
        repo_dpath = dirname(dirname(mmseg.__file__))
        config_dpath = join(repo_dpath, 'configs/_base_/models/vpd_sd.py')
        vpd_cfg = Config.fromfile(config_dpath).stable_diffusion_cfg
        vpd_cfg.pop('checkpoint')

        backbone_cfg = dict(
            type='VPD',
            diffusion_cfg=vpd_cfg,
            class_embed_path='https://download.openmmlab.com/mmsegmentation/'
            'v0.5/vpd/nyu_class_embeddings.pth',
            class_embed_select=True,
            pad_shape=64,
            unet_cfg=dict(use_attn=False),
        )

        head_cfg = dict(
            type='VPDDepthHead',
            max_depth=10,
        )

        self.model = DepthEstimator(
            backbone=backbone_cfg, decode_head=head_cfg)

        inputs = torch.randn(1, 3, 64, 80)
        data_sample = SegDataSample()
        data_sample.gt_depth_map = PixelData(data=torch.rand(1, 64, 80))
        data_sample.set_metainfo(dict(img_shape=(64, 80), ori_shape=(64, 80)))
        self.data = dict(inputs=inputs, data_samples=[data_sample])

    def test_slide_flip_inference(self):

        self.model.test_cfg = ConfigDict(
            dict(mode='slide_flip', crop_size=(64, 64), stride=(16, 16)))

        with torch.no_grad():
            out = self.model.predict(**deepcopy(self.data))

        self.assertEqual(len(out), 1)
        self.assertIn('pred_depth_map', out[0].keys())
        self.assertListEqual(list(out[0].pred_depth_map.shape), [64, 80])

    def test__forward(self):
        data = deepcopy(self.data)
        data['inputs'] = data['inputs'][:, :, :64, :64]
        with torch.no_grad():
            out = self.model._forward(**data)
        self.assertListEqual(list(out.shape), [1, 1, 64, 64])
