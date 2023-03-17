# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import numpy as np
import torch
import torch.nn as nn
from mmengine import ConfigDict
from torch.utils.data import DataLoader, Dataset

from mmseg.apis import MMSegInferencer
from mmseg.models import EncoderDecoder
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


@MODELS.register_module(name='InferExampleHead')
class ExampleDecodeHead(BaseDecodeHead):

    def __init__(self, num_classes=19, out_channels=None):
        super().__init__(
            3, 3, num_classes=num_classes, out_channels=out_channels)

    def forward(self, inputs):
        return self.cls_seg(inputs[0])


@MODELS.register_module(name='InferExampleBackbone')
class ExampleBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        return [self.conv(x)]


@MODELS.register_module(name='InferExampleModel')
class ExampleModel(EncoderDecoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ExampleDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]

    def __getitem__(self, idx):
        return dict(img=torch.tensor([1]), img_metas=dict())

    def __len__(self):
        return 1


def test_inferencer():
    register_all_modules()
    test_dataset = ExampleDataset()
    data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False,
    )

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
        test_dataloader=data_loader)
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
