# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine import ConfigDict
from mmengine.structures import PixelData

from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample
from .utils import _segmentor_forward_train_test


def test_encoder_decoder():

    # test 1 decode head, w.o. aux head

    cfg = ConfigDict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'),
        train_cfg=None,
        test_cfg=dict(mode='whole'))
    segmentor = build_segmentor(cfg)
    _segmentor_forward_train_test(segmentor)

    # test out_channels == 1
    cfg = ConfigDict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(
            type='ExampleDecodeHead', num_classes=2, out_channels=1),
        train_cfg=None,
        test_cfg=dict(mode='whole'))
    segmentor = build_segmentor(cfg)
    _segmentor_forward_train_test(segmentor)

    # test slide mode
    cfg.test_cfg = ConfigDict(mode='slide', crop_size=(3, 3), stride=(2, 2))
    segmentor = build_segmentor(cfg)
    _segmentor_forward_train_test(segmentor)

    # test 1 decode head, 1 aux head
    cfg = ConfigDict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'),
        auxiliary_head=dict(type='ExampleDecodeHead'))
    cfg.test_cfg = ConfigDict(mode='whole')
    segmentor = build_segmentor(cfg)
    _segmentor_forward_train_test(segmentor)

    # test 1 decode head, 2 aux head
    cfg = ConfigDict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'),
        auxiliary_head=[
            dict(type='ExampleDecodeHead'),
            dict(type='ExampleDecodeHead')
        ])
    cfg.test_cfg = ConfigDict(mode='whole')
    segmentor = build_segmentor(cfg)
    _segmentor_forward_train_test(segmentor)


def test_postprocess_result():
    cfg = ConfigDict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'),
        train_cfg=None,
        test_cfg=dict(mode='whole'))
    model = build_segmentor(cfg)

    # test postprocess
    data_sample = SegDataSample()
    data_sample.gt_sem_seg = PixelData(
        **{'data': torch.randint(0, 10, (1, 8, 8))})
    data_sample.set_metainfo({
        'padding_size': (0, 2, 0, 2),
        'ori_shape': (8, 8)
    })
    seg_logits = torch.zeros((1, 2, 10, 10))
    seg_logits[:, :, :8, :8] = 1
    data_samples = [data_sample]

    outputs = model.postprocess_result(seg_logits, data_samples)
    assert outputs[0].seg_logits.data.shape == torch.Size((2, 8, 8))
    assert torch.allclose(outputs[0].seg_logits.data, torch.ones((2, 8, 8)))

    data_sample = SegDataSample()
    data_sample.gt_sem_seg = PixelData(
        **{'data': torch.randint(0, 10, (1, 8, 8))})
    data_sample.set_metainfo({
        'img_padding_size': (0, 2, 0, 2),
        'ori_shape': (8, 8)
    })

    data_samples = [data_sample]
    outputs = model.postprocess_result(seg_logits, data_samples)
    assert outputs[0].seg_logits.data.shape == torch.Size((2, 8, 8))
    assert torch.allclose(outputs[0].seg_logits.data, torch.ones((2, 8, 8)))
