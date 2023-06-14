# Copyright (c) OpenMMLab. All rights reserved.
from os.path import dirname, join

import pytest
import torch
from mmengine import Config
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='skip on cpu due to limited RAM.')
def test_fpn():
    init_default_scope('mmseg')
    repo_dpath = dirname(dirname(dirname(dirname(__file__))))
    cfg = Config.fromfile(
        join(
            repo_dpath,
            'configs/cat_seg/catseg_vitb-r101_4xb2-warmcoslr2e-4-adamw-80k_coco-stuff164k-384x384.py'  # noqa
        ))
    inputs = dict(
        appearance_feat=[
            torch.randn(1, 256, 96, 96),
            torch.randn(1, 512, 48, 48),
            torch.randn(1, 1024, 24, 24)
        ],
        clip_text_feat=torch.randn(171, 80, 512),
        clip_text_feat_test=torch.randn(171, 80, 512),
        clip_img_feat=torch.randn(1, 512, 24, 24),
    )
    cat_aggregator = MODELS.build(cfg.model.neck)
    outputs = cat_aggregator(inputs)
    assert outputs['corr_embed'].shape == torch.Size([1, 128, 171, 24, 24])
    assert outputs['appearance_feats'][0].shape == torch.Size([1, 512, 48, 48])
    assert outputs['appearance_feats'][1].shape == torch.Size([1, 256, 96, 96])
