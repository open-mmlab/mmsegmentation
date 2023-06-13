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
    repo_dpath = dirname(dirname(dirname(dirname(__file__))))
    cfg = Config.fromfile(
        join(
            repo_dpath,
            'configs/cat_seg/catseg_vitb-r101_4xb1-warmcoslr2e-4-adamw-80k_ade20k-384x384.py'  # noqa
        ))
    cfg.model.train_cfg = None
    decode_head = MODELS.build(cfg.model.decode_head)
    inputs = dict(corr_embed=torch.randn(1, 128, 171, 24, 24))
    inputs['appearance_feats'] = (torch.randn(1, 512, 48,
                                              48), torch.randn(1, 256, 96, 96))
    # test inference
    batch_img_metas = [dict(img_shape=(384, 384))]
    test_cfg = cfg.model.test_cfg
    output = decode_head.predict(inputs, batch_img_metas, test_cfg)
    assert output.shape == (1, 171, 384, 384)

    # test training
    inputs = dict(corr_embed=torch.randn(2, 128, 171, 24, 24))
    inputs['appearance_feats'] = (torch.randn(2, 512, 48,
                                              48), torch.randn(2, 256, 96, 96))
    batch_data_samples = []
    img_meta = {
        'img_shape': (384, 384),
        'pad_shape': (384, 384),
    }
    for _ in range(2):
        data_sample = SegDataSample(
            gt_sem_seg=PixelData(data=torch.ones(384, 384).long()))
        data_sample.set_metainfo(img_meta)
        batch_data_samples.append(data_sample)
    train_cfg = {}
    losses = decode_head.loss(inputs, batch_data_samples, train_cfg)
    assert 'loss_ce' in losses.keys()
