# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import DNLHead
from .utils import to_cuda


def test_dnl_head():
    # DNL with 'embedded_gaussian' mode
    head = DNLHead(in_channels=32, channels=16, num_classes=19)
    assert len(head.convs) == 2
    assert hasattr(head, 'dnl_block')
    assert head.dnl_block.temperature == 0.05
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # NonLocal2d with 'dot_product' mode
    head = DNLHead(
        in_channels=32, channels=16, num_classes=19, mode='dot_product')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # NonLocal2d with 'gaussian' mode
    head = DNLHead(
        in_channels=32, channels=16, num_classes=19, mode='gaussian')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # NonLocal2d with 'concatenation' mode
    head = DNLHead(
        in_channels=32, channels=16, num_classes=19, mode='concatenation')
    inputs = [torch.randn(1, 32, 45, 45)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
