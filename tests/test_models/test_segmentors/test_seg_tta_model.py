# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine import ConfigDict
from mmengine.model import BaseTTAModel
from mmengine.structures import PixelData

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
from .utils import *  # noqa: F401,F403

register_all_modules()


def test_encoder_decoder_tta():

    segmentor_cfg = ConfigDict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'),
        train_cfg=None,
        test_cfg=dict(mode='whole'))

    cfg = ConfigDict(type='SegTTAModel', module=segmentor_cfg)

    model: BaseTTAModel = MODELS.build(cfg)

    imgs = []
    data_samples = []
    directions = ['horizontal', 'vertical']
    for i in range(12):
        flip_direction = directions[0] if i % 3 == 0 else directions[1]
        imgs.append(torch.randn(1, 3, 10 + i, 10 + i))
        data_samples.append([
            SegDataSample(
                metainfo=dict(
                    ori_shape=(10, 10),
                    img_shape=(10 + i, 10 + i),
                    flip=(i % 2 == 0),
                    flip_direction=flip_direction),
                gt_sem_seg=PixelData(data=torch.randint(0, 19, (1, 10, 10))))
        ])

    model.test_step(dict(inputs=imgs, data_samples=data_samples))

    # test out_channels == 1
    segmentor_cfg = ConfigDict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(
            type='ExampleDecodeHead',
            num_classes=2,
            out_channels=1,
            threshold=0.4),
        train_cfg=None,
        test_cfg=dict(mode='whole'))
    model.module = MODELS.build(segmentor_cfg)
    for data_sample in data_samples:
        data_sample[0].gt_sem_seg.data = torch.randint(0, 2, (1, 10, 10))
    model.test_step(dict(inputs=imgs, data_samples=data_samples))
