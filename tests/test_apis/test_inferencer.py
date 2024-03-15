# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import numpy as np
import torch
from mmengine import ConfigDict
from utils import *  # noqa: F401, F403

from mmseg.apis import MMSegInferencer
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


def test_inferencer():
    register_all_modules()

    visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer')

    cfg_dict = dict(
        model=dict(
            type='InferExampleModel',
            data_preprocessor=dict(type='SegDataPreProcessor'),
            backbone=dict(type='InferExampleBackbone'),
            decode_head=dict(type='InferExampleHead'),
            test_cfg=dict(mode='whole')),
        visualizer=visualizer,
        test_dataloader=dict(
            dataset=dict(
                type='ExampleDataset',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs')
                ]), ))
    cfg = ConfigDict(cfg_dict)
    model = MODELS.build(cfg.model)

    ckpt = model.state_dict()
    ckpt_filename = tempfile.mktemp()
    torch.save(ckpt, ckpt_filename)

    # test initialization
    infer = MMSegInferencer(cfg, ckpt_filename)

    # test forward
    img = np.random.randint(0, 256, (4, 4, 3))
    infer(img)

    imgs = [img, img]
    infer(imgs)
    results = infer(imgs, out_dir=tempfile.gettempdir())

    # test results
    assert 'predictions' in results
    assert 'visualization' in results
    assert len(results['predictions']) == 2
    assert results['predictions'][0].shape == (4, 4)
