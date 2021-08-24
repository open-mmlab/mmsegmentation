# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import DAHead
from .utils import to_cuda


def test_da_head():

    inputs = [torch.randn(1, 32, 45, 45)]
    head = DAHead(in_channels=32, channels=16, num_classes=19, pam_channels=8)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, tuple) and len(outputs) == 3
    for output in outputs:
        assert output.shape == (1, head.num_classes, 45, 45)
    test_output = head.forward_test(inputs, None, None)
    assert test_output.shape == (1, head.num_classes, 45, 45)
