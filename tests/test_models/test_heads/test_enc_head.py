# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import EncHead
from .utils import to_cuda


def test_enc_head():
    # with se_loss, w.o. lateral
    inputs = [torch.randn(1, 8, 21, 21)]
    head = EncHead(in_channels=[8], channels=4, num_classes=19, in_index=[-1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    assert outputs[0].shape == (1, head.num_classes, 21, 21)
    assert outputs[1].shape == (1, head.num_classes)

    # w.o se_loss, w.o. lateral
    inputs = [torch.randn(1, 8, 21, 21)]
    head = EncHead(
        in_channels=[8],
        channels=4,
        use_se_loss=False,
        num_classes=19,
        in_index=[-1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 21, 21)

    # with se_loss, with lateral
    inputs = [torch.randn(1, 4, 45, 45), torch.randn(1, 8, 21, 21)]
    head = EncHead(
        in_channels=[4, 8],
        channels=4,
        add_lateral=True,
        num_classes=19,
        in_index=[-2, -1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    assert outputs[0].shape == (1, head.num_classes, 21, 21)
    assert outputs[1].shape == (1, head.num_classes)
    test_output = head.forward_test(inputs, None, None)
    assert test_output.shape == (1, head.num_classes, 21, 21)
