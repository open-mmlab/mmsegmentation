# Copyright (c) OpenMMLab. All rights reserved.
from os.path import dirname, join
from unittest import TestCase

import torch
from mmengine import Config

import mmseg
from mmseg.models.backbones import VPD


class TestVPD(TestCase):

    def setUp(self) -> None:

        repo_dpath = dirname(dirname(mmseg.__file__))
        config_dpath = join(repo_dpath, 'configs/_base_/models/vpd_sd.py')
        vpd_cfg = Config.fromfile(config_dpath).stable_diffusion_cfg
        vpd_cfg.pop('checkpoint')

        self.vpd_model = VPD(
            diffusion_cfg=vpd_cfg,
            class_embed_path='https://download.openmmlab.com/mmsegmentation/'
            'v0.5/vpd/nyu_class_embeddings.pth',
            class_embed_select=True,
            pad_shape=64,
            unet_cfg=dict(use_attn=False),
        )

    def test_forward(self):
        # test forward without class_id
        x = torch.randn(1, 3, 60, 60)
        with torch.no_grad():
            out = self.vpd_model(x)

        self.assertEqual(len(out), 4)
        self.assertListEqual(list(out[0].shape), [1, 320, 8, 8])
        self.assertListEqual(list(out[1].shape), [1, 640, 4, 4])
        self.assertListEqual(list(out[2].shape), [1, 1280, 2, 2])
        self.assertListEqual(list(out[3].shape), [1, 1280, 1, 1])

        # test forward with class_id
        x = torch.randn(1, 3, 60, 60)
        with torch.no_grad():
            out = self.vpd_model((x, torch.tensor([2])))

        self.assertEqual(len(out), 4)
        self.assertListEqual(list(out[0].shape), [1, 320, 8, 8])
        self.assertListEqual(list(out[1].shape), [1, 640, 4, 4])
        self.assertListEqual(list(out[2].shape), [1, 1280, 2, 2])
        self.assertListEqual(list(out[3].shape), [1, 1280, 1, 1])
