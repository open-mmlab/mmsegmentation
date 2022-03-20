# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models import Feature2Pyramid


def test_fpn():
    rescales = [4, 2, 1, 0.5]
    embed_dim = 768
    inputs = [torch.randn(1, embed_dim, 32, 32) for i in range(len(rescales))]

    fpn = Feature2Pyramid(embed_dim, rescales, norm='bn')
    outputs = fpn(inputs)
    assert outputs[0].shape == torch.Size([1, 768, 128, 128])
    assert outputs[1].shape == torch.Size([1, 768, 64, 64])
    assert outputs[2].shape == torch.Size([1, 768, 32, 32])
    assert outputs[3].shape == torch.Size([1, 768, 16, 16])
