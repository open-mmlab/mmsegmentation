# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine import Config
from mmengine.registry import init_default_scope

from mmseg.models.text_encoder import CLIPTextEncoder
from mmseg.utils import get_classes


def test_clip_text_encoder():
    init_default_scope('mmseg')
    # test vocabulary
    output_dims = 8
    embed_dims = 32
    vocabulary = ['cat', 'dog', 'bird', 'car', 'bike']
    cfg = dict(
        vocabulary=vocabulary,
        templates=['a photo of a {}.'],
        embed_dims=embed_dims,
        output_dims=output_dims)
    cfg = Config(cfg)

    text_encoder = CLIPTextEncoder(**cfg)
    if torch.cuda.is_available():
        text_encoder = text_encoder.cuda()

    with torch.no_grad():
        class_embeds = text_encoder()
        assert class_embeds.shape == (len(vocabulary) + 1, output_dims)

    # test dataset name
    cfg = dict(
        dataset_name='vaihingen',
        templates=['a photo of a {}.'],
        embed_dims=embed_dims,
        output_dims=output_dims)
    cfg = Config(cfg)

    text_encoder = CLIPTextEncoder(**cfg)
    with torch.no_grad():
        class_embeds = text_encoder()
        class_nums = len(get_classes('vaihingen'))
        assert class_embeds.shape == (class_nums + 1, output_dims)
