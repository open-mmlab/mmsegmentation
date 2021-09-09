# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import ISAHead
from .utils import to_cuda


def test_isa_head():

    inputs = [torch.randn(1, 32, 45, 45)]
    isa_head = ISAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        isa_channels=16,
        down_factor=(8, 8))
    if torch.cuda.is_available():
        isa_head, inputs = to_cuda(isa_head, inputs)
    output = isa_head(inputs)
    assert output.shape == (1, isa_head.num_classes, 45, 45)
