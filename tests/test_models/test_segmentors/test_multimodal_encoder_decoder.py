# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import ConfigDict

from mmseg.models import build_segmentor
from tests.test_models.test_segmentors.utils import \
    _segmentor_forward_train_test


def test_multimodal_encoder_decoder():

    cfg = ConfigDict(
        type='MultimodalEncoderDecoder',
        asymetric_input=False,
        image_encoder=dict(type='ExampleBackbone', out_indices=[1, 2, 3, 4]),
        text_encoder=dict(
            type='ExampleTextEncoder',
            vocabulary=['A', 'B', 'C'],
            output_dims=3),
        decode_head=dict(
            type='ExampleDecodeHead', out_channels=1, num_classes=2),
        train_cfg=None,
        test_cfg=dict(mode='whole'))
    segmentor = build_segmentor(cfg)
    _segmentor_forward_train_test(segmentor)
