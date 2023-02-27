# Copyright (c) OpenMMLab. All rights reserved.
from os.path import dirname, join

import torch
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.structures import PixelData

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


def test_maskformer_head():
    init_default_scope('mmseg')
    repo_dpath = dirname(dirname(__file__))
    cfg = Config.fromfile(
        join(
            repo_dpath,
            '../../configs/maskformer/maskformer_r50-d32_8xb2-160k_ade20k-512x512.py'  # noqa
        ))
    cfg.model.train_cfg = None
    decode_head = MODELS.build(cfg.model.decode_head)
    inputs = (torch.randn(1, 256, 32, 32), torch.randn(1, 512, 16, 16),
              torch.randn(1, 1024, 8, 8), torch.randn(1, 2048, 4, 4))
    # test inference
    batch_img_metas = [
        dict(
            scale_factor=(1.0, 1.0),
            img_shape=(512, 683),
            ori_shape=(512, 683))
    ]
    test_cfg = dict(mode='whole')
    output = decode_head.predict(inputs, batch_img_metas, test_cfg)
    assert output.shape == (1, 150, 512, 683)

    # test training
    inputs = (torch.randn(2, 256, 32, 32), torch.randn(2, 512, 16, 16),
              torch.randn(2, 1024, 8, 8), torch.randn(2, 2048, 4, 4))
    batch_data_samples = []
    img_meta = {
        'img_shape': (512, 512),
        'ori_shape': (480, 640),
        'pad_shape': (512, 512),
        'scale_factor': (1.425, 1.425),
    }
    for _ in range(2):
        data_sample = SegDataSample(
            gt_sem_seg=PixelData(data=torch.ones(512, 512).long()))
        data_sample.set_metainfo(img_meta)
        batch_data_samples.append(data_sample)
    train_cfg = {}
    losses = decode_head.loss(inputs, batch_data_samples, train_cfg)
    assert (loss in losses.keys()
            for loss in ('loss_cls', 'loss_mask', 'loss_dice'))
