# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.decode_heads import CCHead
from .utils import to_cuda


def test_cc_head():
    head = CCHead(in_channels=32, channels=16, num_classes=19)
    assert len(head.convs) == 2
    assert hasattr(head, 'cca')
    if not torch.cuda.is_available():
        pytest.skip('CCHead requires CUDA')
    inputs = [torch.randn(1, 32, 45, 45)]
    head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
