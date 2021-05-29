from mmcv import ConfigDict

from mmseg.models import build_segmentor
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
